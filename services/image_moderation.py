"""
Image Moderation Service - IMPROVED VERSION
Perbaikan: Violence detection lebih akurat, mengurangi false positive

PERUBAHAN:
1. Violence threshold dinaikkan ke 0.85 (dari 0.75)
2. Tambah ensemble voting untuk violence (butuh 2 indikator)
3. Tambah context awareness untuk gambar normal
4. Improve logging dan scoring
"""

import torch
import logging
import os
from PIL import Image
from typing import Dict, Tuple, List
import io
import base64
import requests
from dataclasses import dataclass
import traceback
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

from transformers import (
    AutoModelForImageClassification,
    ViTImageProcessor,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_ultralytics_yolo(model_path: str):
    """Load YOLOv8 model via ultralytics."""
    try:
        from ultralytics import YOLO
        return YOLO(model_path)
    except ImportError:
        raise ImportError("ultralytics belum di-install. Jalankan: pip install ultralytics")


@dataclass
class ModerationResult:
    """Data class for moderation results"""
    status: str
    decision: str
    flagged: bool
    scores: Dict
    categories: Dict
    detection_reasons: list
    extracted_image_url: str = None

    def to_dict(self):
        result = {
            'status': self.status,
            'decision': self.decision,
            'flagged': self.flagged,
            'scores': self.scores,
            'categories': self.categories,
            'detection_reasons': self.detection_reasons
        }
        if self.extracted_image_url:
            result['extracted_image_url'] = self.extracted_image_url
        return result


# Adult domain blocklist
ADULT_DOMAINS = [
    "pornhub.com", "xvideos.com", "xnxx.com", "redtube.com", "youporn.com",
    "tube8.com", "hardsex.com", "spankbang.com", "hclips.com", "hd18.xxx",
    "brazzers.com", "momporn.xxx", "beeg.com", "eporner.com", "naughtyamerica.com",
    "digitalplayground.com", "badgirlsporn.com", "bangbros.com", "blpcollection.com",
    "chaturbate.com", "camsin.com", "stripchat.com", "myfreecams.com",
    "xhamster.com", "pornxxx.net", "xxx.com",
]

# Trusted news/media domains
TRUSTED_NEWS_DOMAINS = [
    "detik.com", "kompas.com", "tribunnews.com", "liputan6.com", "tempo.co",
    "cnnindonesia.com", "antaranews.com", "republika.co.id", "sindonews.com",
    "okezone.com", "jpnn.com", "suara.com", "merdeka.com", "viva.co.id",
    "bbc.com", "cnn.com", "reuters.com", "apnews.com", "nytimes.com",
    "theguardian.com", "washingtonpost.com", "aljazeera.com", "bloomberg.com",
    "forbes.com", "time.com", "newsweek.com", "nbcnews.com", "abcnews.go.com",
    "tni.mil.id", "polri.go.id", "kemhan.go.id", "defense.gov", "army.mil",
    "substack.com", "substackcdn.com", "medium.com", "blogger.com", "wordpress.com",  # TAMBAH platform blogging
]


def _is_adult_domain(url: str) -> bool:
    """Check if URL is from adult domain blocklist"""
    try:
        hostname = urlparse(url).hostname or ""
        hostname = hostname.lstrip("www.")
        return any(hostname == d or hostname.endswith("." + d) for d in ADULT_DOMAINS)
    except Exception:
        return False


def _is_trusted_news_domain(url: str) -> bool:
    """Check if URL is from trusted news/media domain"""
    try:
        hostname = urlparse(url).hostname or ""
        hostname = hostname.lstrip("www.")
        return any(hostname == d or hostname.endswith("." + d) for d in TRUSTED_NEWS_DOMAINS)
    except Exception:
        return False


class ImageModerationService:

    def __init__(
        self,
        primary_nsfw_model: str = "Falconsai/nsfw_image_detection",
        detailed_nsfw_model: str = "LukeJacob2023/nsfw-image-detector",  # TETAP pakai yang lama
        violence_model: str = "jaranohaal/vit-base-violence-detection",
        weapon_model_path: str = "weights/firearm_best.pt",
        device: str = None,
        nsfw_threshold: float = 0.55,
        detailed_nsfw_threshold: float = 0.50,
        violence_threshold: float = 0.70,  # Turun dari 0.85 untuk detect gore
        weapon_confidence_threshold: float = 0.45,
        armed_person_violence_threshold: float = 0.90,
        armed_max_persons: int = 3,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.nsfw_threshold = nsfw_threshold
        self.detailed_nsfw_threshold = detailed_nsfw_threshold
        self.violence_threshold = violence_threshold
        self.weapon_confidence_threshold = weapon_confidence_threshold
        self.armed_person_violence_threshold = armed_person_violence_threshold
        self.armed_max_persons = armed_max_persons

        logger.info(f"🚀 Initializing ImageModerationService (IMPROVED - LOWER FALSE POSITIVE)")
        logger.info(f"📊 Thresholds:")
        logger.info(f"   NSFW: {nsfw_threshold}")
        logger.info(f"   Violence: {violence_threshold} (IMPROVED)")
        logger.info(f"   Weapon: {weapon_confidence_threshold}")

        try:
            logger.info("📦 Loading primary NSFW model...")
            self.primary_processor = ViTImageProcessor.from_pretrained(primary_nsfw_model)
            self.primary_model = AutoModelForImageClassification.from_pretrained(
                primary_nsfw_model
            ).to(self.device)
            self.primary_model.eval()

            logger.info("📦 Loading detailed NSFW classifier...")
            self.detailed_processor = ViTImageProcessor.from_pretrained(detailed_nsfw_model)
            self.detailed_model = AutoModelForImageClassification.from_pretrained(
                detailed_nsfw_model
            ).to(self.device)
            self.detailed_model.eval()

            logger.info("📦 Loading violence detector...")
            self.violence_processor = ViTImageProcessor.from_pretrained(violence_model)
            self.violence_model = AutoModelForImageClassification.from_pretrained(
                violence_model
            ).to(self.device)
            self.violence_model.eval()

            logger.info("📦 Loading firearm detector (YOLOv8n)...")
            
            if not os.path.exists(weapon_model_path):
                logger.warning(f"⚠️ Weapon model not found at: {weapon_model_path}")
                logger.info("📥 Auto-downloading from Hugging Face...")
                try:
                    from huggingface_hub import hf_hub_download
                    os.makedirs(os.path.dirname(weapon_model_path) or "weights", exist_ok=True)
                    
                    downloaded_path = hf_hub_download(
                        repo_id="Subh775/Firearm_Detection_Yolov8n",
                        filename="weights/best.pt",
                    )
                    
                    import shutil
                    shutil.copy(downloaded_path, weapon_model_path)
                    logger.info(f"✅ Model downloaded to: {weapon_model_path}")
                except Exception as download_error:
                    logger.error(f"❌ Failed to auto-download: {download_error}")
            
            try:
                self.weapon_yolo = _load_ultralytics_yolo(weapon_model_path)
                if self.device == "cuda":
                    self.weapon_yolo.to("cuda")
                self.weapon_enabled = True
                logger.info("✅ Firearm detector loaded!")
            except Exception as e:
                logger.error(f"❌ Firearm detector failed: {e}")
                self.weapon_enabled = False
                self.weapon_yolo = None

            logger.info("✅ All models loaded!")

        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            logger.error(traceback.format_exc())
            raise

    def _extract_image_from_html(self, html_content: bytes, base_url: str) -> str:
        """Extract image URL from HTML page"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            og_image = soup.find('meta', property='og:image')
            if og_image and og_image.get('content'):
                return urljoin(base_url, og_image['content'])

            twitter_image = soup.find('meta', attrs={'name': 'twitter:image'})
            if twitter_image and twitter_image.get('content'):
                return urljoin(base_url, twitter_image['content'])

            img_candidates = []
            for img in soup.find_all('img'):
                src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
                if not src or src.startswith('data:'):
                    continue

                width = img.get('width', '0')
                height = img.get('height', '0')
                try:
                    if width.isdigit() and height.isdigit():
                        if int(width) < 200 or int(height) < 200:
                            continue
                except (ValueError, TypeError):
                    pass

                if any(p in src.lower() for p in ['icon', 'logo', 'sprite', 'banner']):
                    continue

                img_candidates.append(urljoin(base_url, src))

            if img_candidates:
                return img_candidates[0]

            raise ValueError("No suitable image found in HTML")

        except Exception as e:
            logger.error(f"Error extracting image: {e}")
            raise

    def _load_image(self, image_data: str, image_type: str = 'base64') -> Image.Image:
        """Load image from base64 or URL"""
        try:
            is_data_uri = image_data.startswith('data:image/')
            is_http_url = image_data.startswith('http://') or image_data.startswith('https://')

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            }

            if is_http_url:
                response = requests.get(image_data, timeout=10, headers=headers)
                response.raise_for_status()

                content_type = response.headers.get('Content-Type', '').lower()

                if 'text/html' in content_type:
                    image_url = self._extract_image_from_html(response.content, image_data)
                    img_response = requests.get(image_url, timeout=10, headers=headers)
                    img_response.raise_for_status()
                    image = Image.open(io.BytesIO(img_response.content))
                    self._extracted_image_url = image_url
                else:
                    image = Image.open(io.BytesIO(response.content))
                    self._extracted_image_url = image_data

            elif is_data_uri or image_type == 'base64':
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))

            else:
                response = requests.get(image_data, timeout=10, headers=headers)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content))

            if image.mode != 'RGB':
                image = image.convert('RGB')

            return image

        except Exception as e:
            logger.error(f"❌ Error loading image: {e}")
            raise

    def _check_primary_nsfw(self, image: Image.Image) -> Tuple[bool, float, str, Dict]:
        """Primary NSFW detection"""
        try:
            inputs = self.primary_processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.primary_model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

            predicted_idx = outputs.logits.argmax(-1).item()
            confidence = probabilities[0][predicted_idx].item()
            category = self.primary_model.config.id2label[predicted_idx]

            probs_dict = {
                label: probabilities[0][idx].item()
                for idx, label in self.primary_model.config.id2label.items()
            }

            is_nsfw = category.lower() == "nsfw" and confidence >= self.nsfw_threshold
            logger.info(f"📸 Primary NSFW: {category} ({confidence:.3f}) {'🔴' if is_nsfw else '🟢'}")
            return is_nsfw, confidence, category, probs_dict

        except Exception as e:
            logger.error(f"Error in primary NSFW: {e}")
            return False, 0.0, "error", {}

    def _check_detailed_nsfw(self, image: Image.Image) -> Tuple[bool, float, str, Dict]:
        """Detailed NSFW classification"""
        try:
            inputs = self.detailed_processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.detailed_model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

            predicted_idx = outputs.logits.argmax(-1).item()
            confidence = probabilities[0][predicted_idx].item()
            category = self.detailed_model.config.id2label[predicted_idx]

            probs_dict = {
                label: probabilities[0][idx].item()
                for idx, label in self.detailed_model.config.id2label.items()
            }

            nsfw_categories = ['porn', 'hentai', 'sexy']
            is_nsfw = False

            if category in nsfw_categories and confidence >= self.detailed_nsfw_threshold:
                is_nsfw = True
            elif any(probs_dict.get(cat, 0) >= self.detailed_nsfw_threshold for cat in nsfw_categories):
                is_nsfw = True
                category = max(nsfw_categories, key=lambda x: probs_dict.get(x, 0))
                confidence = probs_dict[category]

            logger.info(f"🔞 Detailed NSFW: {category} ({confidence:.3f}) {'🔴' if is_nsfw else '🟢'}")
            return is_nsfw, confidence, category, probs_dict

        except Exception as e:
            logger.error(f"Error in detailed NSFW: {e}")
            return False, 0.0, "error", {}

    def _check_violence(self, image: Image.Image) -> Tuple[bool, float, str, Dict]:
        """Violence detection - IMPROVED dengan threshold lebih rendah untuk gore"""
        try:
            inputs = self.violence_processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.violence_model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

            predicted_idx = outputs.logits.argmax(-1).item()
            confidence = probabilities[0][predicted_idx].item()
            category = self.violence_model.config.id2label[predicted_idx]

            probs_dict = {
                label: probabilities[0][idx].item()
                for idx, label in self.violence_model.config.id2label.items()
            }

            violence_prob = probs_dict.get('violence', probs_dict.get('LABEL_0', 0))
            normal_prob = probs_dict.get('normal', probs_dict.get('LABEL_1', 0))
            
            # Check violence dengan threshold
            is_violent = violence_prob >= self.violence_threshold
            
            logger.info(f"⚔️  Violence: {category} ({confidence:.3f}) {'🔴' if is_violent else '🟢'}")
            logger.info(f"   Violence:{violence_prob:.2f} Normal:{normal_prob:.2f}")

            return is_violent, confidence, category, probs_dict

        except Exception as e:
            logger.error(f"Error in violence check: {e}")
            return False, 0.0, "error", {}

    def _check_weapons(self, image: Image.Image) -> Tuple[bool, float, List[Dict], int]:
        """Weapon detection using YOLOv8 + person counting"""
        if not self.weapon_enabled:
            logger.info("🔫 Weapons: Detector not available")
            return False, 0.0, [], 0

        try:
            results = self.weapon_yolo.predict(
                source=image,
                conf=self.weapon_confidence_threshold,
                device=0 if self.device == "cuda" else "cpu",
                verbose=False,
            )

            boxes = results[0].boxes
            weapon_detections: List[Dict] = []
            person_count = 0

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                label = self.weapon_yolo.names.get(cls_id, f"class_{cls_id}")
                box = boxes.xyxy[i].tolist()

                if label.lower() == 'person' or cls_id == 0:
                    person_count += 1
                else:
                    weapon_detections.append({
                        'label': label,
                        'confidence': conf,
                        'box': box,
                    })

            has_weapons = len(weapon_detections) > 0
            max_confidence = max((d['confidence'] for d in weapon_detections), default=0.0)

            if has_weapons:
                logger.info(f"🔫 Weapons: {len(weapon_detections)} detected 🔴")
                for det in weapon_detections[:5]:
                    logger.info(f"   {det['label']} ({det['confidence']:.3f})")
            else:
                logger.info("🔫 Weapons: None detected 🟢")
            
            if person_count > 0:
                logger.info(f"👥 Persons: {person_count} detected")

            return has_weapons, max_confidence, weapon_detections, person_count

        except Exception as e:
            logger.error(f"Error in weapon detection: {e}")
            return False, 0.0, [], 0

    def moderate(self, image_data: str, image_type: str = 'base64') -> ModerationResult:
        """Main moderation function - IMPROVED VERSION"""

        # Check adult domain blocklist
        if image_type in ('url',) or image_data.startswith('http://') or image_data.startswith('https://'):
            if _is_adult_domain(image_data):
                logger.info(f"🚫 BLOCKED: Adult domain — {image_data}")
                return ModerationResult(
                    status='success',
                    decision='DITOLAK',
                    flagged=True,
                    scores={'nsfw': 1.0, 'violence': 0, 'weapon': 0},
                    categories={'nsfw': True, 'adult_domain': True},
                    detection_reasons=[f"Domain dewasa terdeteksi: {urlparse(image_data).hostname}"],
                    extracted_image_url=None,
                )
            
            # Check trusted domain - auto bypass detailed NSFW false positive
            is_trusted_domain = _is_trusted_news_domain(image_data)
            if is_trusted_domain:
                logger.info(f"✅ TRUSTED DOMAIN: {urlparse(image_data).hostname}")
        else:
            is_trusted_domain = False

        try:
            logger.info("=" * 80)
            logger.info("🔍 IMAGE MODERATION (IMPROVED)")
            logger.info("=" * 80)

            image = self._load_image(image_data, image_type)

            # Run ALL checks
            primary_nsfw, primary_conf, primary_cat, primary_probs = self._check_primary_nsfw(image)
            detailed_nsfw, detailed_conf, detailed_cat, detailed_probs = self._check_detailed_nsfw(image)
            is_violent, violence_conf, violence_cat, violence_probs = self._check_violence(image)
            has_weapons, weapon_conf, weapon_detections, person_count = self._check_weapons(image)

            violence_prob = violence_probs.get('violence', violence_probs.get('LABEL_0', 0))
            normal_prob = violence_probs.get('normal', violence_probs.get('LABEL_1', 0))
            nsfw_score = max(primary_conf if primary_nsfw else 0, detailed_conf if detailed_nsfw else 0)

            # VERY STRICT military bypass - only for truly official photos
            is_military_context = False
            if has_weapons and person_count >= 15 and violence_prob < 0.15 and nsfw_score < 0.10:
                is_military_context = True
                logger.info(f"✅ MILITARY/POLICE UNIT (VERY STRICT):")
                logger.info(f"   Persons: {person_count}, Violence: {violence_prob:.3f}, NSFW: {nsfw_score:.3f}")

            # Build result
            categories: Dict = {}
            detection_reasons: List[str] = []
            flagged = False

            # NSFW - IMPROVED ENSEMBLE VOTING
            # Rule 1: Detailed detect PORN atau SEXY = langsung REJECT (bukan kartun)
            # Rule 2: Primary + Detailed setuju = REJECT
            # Rule 3: Detailed detect HENTAI tapi primary bilang normal = cek context
            
            if detailed_nsfw and detailed_cat in ['porn', 'sexy']:
                # Porn/Sexy = konten dewasa nyata, bukan kartun
                flagged = True
                categories['nsfw'] = True
                categories[f'nsfw_{detailed_cat}'] = True
                detection_reasons.append(f"NSFW Content: {detailed_cat} ({detailed_conf:.2f})")
                
            elif primary_nsfw and detailed_nsfw:
                # Kedua model setuju
                flagged = True
                categories['nsfw'] = True
                detection_reasons.append(f"NSFW Content: {primary_cat} ({primary_conf:.2f}), {detailed_cat} ({detailed_conf:.2f})")
                categories[f'nsfw_{detailed_cat}'] = True
                
            elif primary_nsfw and not detailed_nsfw:
                # Primary saja detect, butuh confidence tinggi
                if primary_conf >= 0.85:
                    flagged = True
                    categories['nsfw'] = True
                    detection_reasons.append(f"NSFW Content: {primary_cat} ({primary_conf:.2f}) - High confidence")
                else:
                    logger.info(f"⚪ Primary NSFW {primary_conf:.2f} tidak cukup tinggi tanpa detailed confirmation")
                    
            elif detailed_nsfw and detailed_cat == 'hentai' and not primary_nsfw:
                # Detailed detect hentai tapi primary bilang normal = kemungkinan kartun
                # Override jika dari trusted domain
                if is_trusted_domain:
                    logger.info(f"⚪ Detailed NSFW {detailed_cat} di-override (trusted domain: {urlparse(image_data).hostname})")
                else:
                    # Bukan trusted domain, tapi bisa jadi kartun biasa
                    # Cek confidence detailed - jika sangat tinggi tetap flag
                    if detailed_conf >= 0.95:
                        flagged = True
                        categories['nsfw'] = True
                        categories[f'nsfw_{detailed_cat}'] = True
                        detection_reasons.append(f"NSFW Content: {detailed_cat} ({detailed_conf:.2f}) - Very high confidence")
                    else:
                        logger.info(f"⚪ Detailed NSFW {detailed_cat} ({detailed_conf:.2f}) di-ignore (kemungkinan kartun/animasi)")

            # Violence - Direct check
            if is_violent:
                flagged = True
                categories['violence'] = True
                detection_reasons.append(f"Violence/Gore Detected ({violence_prob:.2f})")

            # Weapons
            if has_weapons:
                if is_military_context:
                    logger.info(f"⚪ Weapons bypassed (military context)")
                    categories['military_police'] = True
                else:
                    flagged = True
                    categories['weapons'] = True
                    weapon_list = ', '.join(f"{d['label']}({d['confidence']:.2f})" for d in weapon_detections[:5])
                    detection_reasons.append(f"Weapons: {weapon_list}")

            # Scores untuk frontend - HANYA tampilkan yang relevan
            scores = {
                'nsfw': nsfw_score if flagged and 'nsfw' in categories else 0,  # Hide jika tidak flagged NSFW
                'violence': violence_prob if flagged and 'violence' in categories else 0,  # Hide jika tidak flagged
                'weapon': weapon_conf if has_weapons else 0,
                'primary': primary_probs,
                'detailed': detailed_probs,
                'violence_probs': violence_probs,
            }

            decision = 'DITOLAK' if flagged else 'AMAN'

            logger.info("=" * 80)
            logger.info(f"📊 RESULT: {decision}")
            if flagged:
                for reason in detection_reasons:
                    logger.info(f"   • {reason}")
            else:
                logger.info(f"   ✅ Gambar aman, semua checks passed")
            logger.info("=" * 80)

            return ModerationResult(
                status='success',
                decision=decision,
                flagged=flagged,
                scores=scores,
                categories=categories,
                detection_reasons=detection_reasons,
                extracted_image_url=getattr(self, '_extracted_image_url', None),
            )

        except Exception as e:
            logger.error(f"❌ Moderation error: {e}")
            logger.error(traceback.format_exc())
            return ModerationResult(
                status='error',
                decision='DITOLAK',
                flagged=True,
                scores={},
                categories={'error': True},
                detection_reasons=[f"Gagal memproses gambar: {e}"],
                extracted_image_url=None,
            )


if __name__ == "__main__":
    service = ImageModerationService()
    print("✅ ImageModerationService initialized (IMPROVED VERSION)!")