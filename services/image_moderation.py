import requests
import os
import base64
import logging
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImageModerationService:
    def __init__(self):
        self.api_user = os.getenv('SIGHTENGINE_API_USER')
        self.api_secret = os.getenv('SIGHTENGINE_API_SECRET')
        self.base_url = 'https://api.sightengine.com/1.0/check.json'
        logger.info("ImageModerationService initialized: Sightengine only")

    # ========================================
    # SIGHTENGINE API CALL
    # ========================================
    def _call_sightengine(self, image_data, image_type='base64') -> dict:
        params = {
            'models': 'nudity-2.1,gore-2.0,offensive,text-content,weapon,recreational_drug,medical,alcohol,tobacco,scam,violence',
            'api_user': self.api_user,
            'api_secret': self.api_secret
        }

        if image_type == 'url':
            params['url'] = image_data
            response = requests.get(self.base_url, params=params)
        else:
            image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
            files = {'media': image_bytes}
            response = requests.post(self.base_url, data=params, files=files)

        response.raise_for_status()
        return response.json()

    # ========================================
    # SCORE ANALYSIS — 3-TIER DECISION
    # ========================================
    def _analyze_scores(self, result: dict) -> dict:
        """
        Parse Sightengine response → structured scores + 3-tier decision.

        Response structure dari Sightengine (CONFIRMED dari docs):
          nudity            → { sexual_activity, sexual_display, erotica, sextoy, suggestive, raw, partial }
          gore              → { prob, classes: {...}, type: {...} }
          offensive         → { prob, classes: {...} }
          weapon            → { classes: { firearm, firearm_gesture, firearm_toy, knife, ... } }  ← NO .prob!
          violence          → { prob, classes: {...} }
          recreational_drug → { prob, classes: {...} }
          medical           → { prob, classes: {...} }
          alcohol           → { prob, classes: {...} }
          tobacco           → { prob, classes: {...} }
          scam              → { prob, classes: {...} }
        """
        scores = {}
        categories = {}
        tier = 'AMAN'

        # ========================================
        # NUDITY
        # ========================================
        if 'nudity' in result:
            nudity = result['nudity']

            sexual_activity = nudity.get('sexual_activity', 0)
            sexual_display  = nudity.get('sexual_display', 0)
            erotica         = nudity.get('erotica', 0)
            sextoy          = nudity.get('sextoy', 0)
            suggestive      = nudity.get('suggestive', 0)
            raw_score       = nudity.get('raw', 0)
            partial_score   = nudity.get('partial', 0)

            scores['nudity'] = {
                'sexual_activity': sexual_activity,
                'sexual_display': sexual_display,
                'erotica': erotica,
                'sextoy': sextoy,
                'suggestive': suggestive,
                'raw': raw_score,
                'partial': partial_score
            }

            # DITOLAK — eksplisit
            if (sexual_activity > 0.6 or
                sexual_display > 0.6 or
                erotica > 0.7 or
                sextoy > 0.6 or
                (sexual_activity > 0.4 and sexual_display > 0.4) or
                (raw_score > 0.8 and sexual_display > 0.3)):

                tier = 'DITOLAK'
                categories['nudity'] = True
                logger.info("🔴 Nudity: DITOLAK")

            # PERLU_DITINJAU — borderline
            elif (sexual_activity > 0.25 or
                  sexual_display > 0.25 or
                  erotica > 0.35 or
                  sextoy > 0.3 or
                  (raw_score > 0.6 and suggestive > 0.3) or
                  (partial_score > 0.5 and suggestive > 0.4)):

                if tier == 'AMAN':
                    tier = 'PERLU_DITINJAU'
                categories['nudity_suspicious'] = True
                logger.info("🟡 Nudity: PERLU_DITINJAU")

            else:
                logger.info("🟢 Nudity: AMAN")

        # ========================================
        # GORE — response: { prob, classes, type }
        # ========================================
        if 'gore' in result:
            score = result['gore'].get('prob', 0)
            scores['gore'] = score

            if score > 0.7:
                tier = 'DITOLAK'
                categories['gore'] = True
                logger.info("🔴 Gore: DITOLAK")
            elif score > 0.4:
                if tier == 'AMAN':
                    tier = 'PERLU_DITINJAU'
                categories['gore_suspicious'] = True
                logger.info("🟡 Gore: PERLU_DITINJAU")

        # ========================================
        # OFFENSIVE — response: { prob, classes }
        # ========================================
        if 'offensive' in result:
            score = result['offensive'].get('prob', 0)
            scores['offensive'] = score

            if score > 0.7:
                tier = 'DITOLAK'
                categories['offensive'] = True
                logger.info("🔴 Offensive: DITOLAK")
            elif score > 0.5:
                if tier == 'AMAN':
                    tier = 'PERLU_DITINJAU'
                categories['offensive_suspicious'] = True
                logger.info("🟡 Offensive: PERLU_DITINJAU")

        # ========================================
        # WEAPON — response: { classes: { firearm, knife, ... } }
        # ⚠️  weapon TIDAK punya .prob — ambil max dari classes
        # ========================================
        if 'weapon' in result:
            weapon_data = result['weapon']
            classes = weapon_data.get('classes', {})
            # Ambil score tertinggi dari semua weapon classes
            score = max(classes.values()) if classes else 0
            scores['weapon'] = score
            scores['weapon_classes'] = classes

            if score > 0.7:
                tier = 'DITOLAK'
                categories['weapon'] = True
                logger.info(f"🔴 Weapon: DITOLAK (max class: {score:.4f})")
            elif score > 0.5:
                if tier == 'AMAN':
                    tier = 'PERLU_DITINJAU'
                categories['weapon_suspicious'] = True
                logger.info(f"🟡 Weapon: PERLU_DITINJAU (max class: {score:.4f})")

        # ========================================
        # RECREATIONAL DRUG — response: { prob, classes }
        # ========================================
        if 'recreational_drug' in result:
            score = result['recreational_drug'].get('prob', 0)
            scores['recreational_drug'] = score

            if score > 0.7:
                tier = 'DITOLAK'
                categories['recreational_drug'] = True
                logger.info("🔴 Recreational Drug: DITOLAK")
            elif score > 0.5:
                if tier == 'AMAN':
                    tier = 'PERLU_DITINJAU'
                categories['recreational_drug_suspicious'] = True
                logger.info("🟡 Recreational Drug: PERLU_DITINJAU")

        # ========================================
        # MEDICAL — response: { prob, classes }
        # Only flag kalau kombinasi dengan recreational_drug
        # ========================================
        if 'medical' in result:
            score = result['medical'].get('prob', 0)
            scores['medical'] = score

            if scores.get('recreational_drug', 0) > 0.3 and score > 0.5:
                if tier == 'AMAN':
                    tier = 'PERLU_DITINJAU'
                categories['medical_suspicious'] = True
                logger.info("🟡 Medical (+ rec drug): PERLU_DITINJAU")

        # ========================================
        # VIOLENCE — response: { prob, classes }
        # ========================================
        if 'violence' in result:
            score = result['violence'].get('prob', 0)
            scores['violence'] = score

            if score > 0.7:
                tier = 'DITOLAK'
                categories['violence'] = True
                logger.info("🔴 Violence: DITOLAK")
            elif score > 0.5:
                if tier == 'AMAN':
                    tier = 'PERLU_DITINJAU'
                categories['violence_suspicious'] = True
                logger.info("🟡 Violence: PERLU_DITINJAU")

        # ========================================
        # ALCOHOL — response: { prob, classes }
        # ========================================
        if 'alcohol' in result:
            score = result['alcohol'].get('prob', 0)
            scores['alcohol'] = score

            if score > 0.7:
                tier = 'DITOLAK'
                categories['alcohol'] = True
                logger.info("🔴 Alcohol: DITOLAK")
            elif score > 0.5:
                if tier == 'AMAN':
                    tier = 'PERLU_DITINJAU'
                categories['alcohol_suspicious'] = True
                logger.info("🟡 Alcohol: PERLU_DITINJAU")

        # ========================================
        # TOBACCO — response: { prob, classes }
        # ========================================
        if 'tobacco' in result:
            score = result['tobacco'].get('prob', 0)
            scores['tobacco'] = score

            if score > 0.7:
                tier = 'DITOLAK'
                categories['tobacco'] = True
                logger.info("🔴 Tobacco: DITOLAK")
            elif score > 0.5:
                if tier == 'AMAN':
                    tier = 'PERLU_DITINJAU'
                categories['tobacco_suspicious'] = True
                logger.info("🟡 Tobacco: PERLU_DITINJAU")

        # ========================================
        # SCAM — response: { prob, classes }
        # ========================================
        if 'scam' in result:
            score = result['scam'].get('prob', 0)
            scores['scam'] = score

            if score > 0.7:
                tier = 'DITOLAK'
                categories['scam'] = True
                logger.info("🔴 Scam: DITOLAK")
            elif score > 0.5:
                if tier == 'AMAN':
                    tier = 'PERLU_DITINJAU'
                categories['scam_suspicious'] = True
                logger.info("🟡 Scam: PERLU_DITINJAU")

        return {
            'scores': scores,
            'categories': categories,
            'tier': tier
        }

    # ========================================
    # MAIN ENTRY POINT
    # ========================================
    def moderate(self, image_data, image_type='base64') -> dict:
        """
        3-tier moderation — Sightengine only.

        AMAN           → semua scores rendah, langsung approve
        PERLU_DITINJAU → ada score borderline, perlu review manual
        DITOLAK        → score tinggi, langsung reject
        """
        try:
            logger.info("📷 Sightengine scanning...")
            raw_result = self._call_sightengine(image_data, image_type)

            analysis = self._analyze_scores(raw_result)
            tier = analysis['tier']

            moderation_result = {
                'status': 'success',
                'decision': tier,
                'flagged': tier == 'DITOLAK',
                'needs_review': tier == 'PERLU_DITINJAU',
                'categories': analysis['categories'],
                'scores': analysis['scores'],
                'raw_response': raw_result
            }

            if tier == 'DITOLAK':
                logger.info(f"🚫 DITOLAK | Flagged: {list(analysis['categories'].keys())}")
            elif tier == 'PERLU_DITINJAU':
                logger.info(f"⚠️  PERLU_DITINJAU | Suspicious: {list(analysis['categories'].keys())}")
            else:
                logger.info("✅ AMAN")

            return moderation_result

        except Exception as e:
            # FIX: tambahkan exc_info=True supaya full traceback tercetak
            logger.error(f"Moderation error: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e)
            }


# ========================================
# TESTING
# ========================================
if __name__ == '__main__':
    os.environ['SIGHTENGINE_API_USER'] = 'GANTI_DENGAN_API_USER_ANDA'
    os.environ['SIGHTENGINE_API_SECRET'] = 'GANTI_DENGAN_API_SECRET_ANDA'

    service = ImageModerationService()
    test_url = 'https://example.com/test.jpg'

    print("=" * 70)
    print("IMAGE MODERATION — Sightengine 3-Tier")
    print("=" * 70)

    result = service.moderate(test_url, image_type='url')

    print(f"\nStatus: {result['status']}")
    print(f"\n{'=' * 70}")
    print(f"DECISION: {result.get('decision', 'N/A')}")
    print(f"{'=' * 70}")

    if result.get('decision') == 'DITOLAK':
        print("\n🚫 DITOLAK - Konten berbahaya terdeteksi")
    elif result.get('decision') == 'PERLU_DITINJAU':
        print("\n⚠️  PERLU DITINJAU - Perlu review manual")
    else:
        print("\n✅ AMAN - Tidak ada konten berbahaya terdeteksi")

    # Nudity scores
    if 'nudity' in result.get('scores', {}):
        print(f"\n--- NUDITY SCORES ---")
        for key, value in result['scores']['nudity'].items():
            indicator = "🔴" if value > 0.6 else ("🟡" if value > 0.25 else "🟢")
            print(f"  {indicator} {key:20s}: {value:.4f}")

    # Other scores
    print(f"\n--- OTHER SCORES ---")
    for category, score in result.get('scores', {}).items():
        if category not in ('nudity', 'weapon_classes') and isinstance(score, (int, float)):
            indicator = "🔴" if score > 0.7 else ("🟡" if score > 0.4 else "🟢")
            print(f"  {indicator} {category:20s}: {score:.4f}")

    # Weapon detail
    if 'weapon_classes' in result.get('scores', {}):
        print(f"\n--- WEAPON CLASSES ---")
        for cls, val in result['scores']['weapon_classes'].items():
            indicator = "🔴" if val > 0.7 else ("🟡" if val > 0.4 else "🟢")
            print(f"  {indicator} {cls:25s}: {val:.4f}")

    # Flagged
    if result.get('categories'):
        print(f"\n--- FLAGGED CATEGORIES ---")
        for cat, flagged in result['categories'].items():
            if flagged:
                print(f"  ⚑ {cat}")