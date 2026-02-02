from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from dotenv import load_dotenv
import os
import sys
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Clear import cache for image_moderation (temporary fix)
if 'services.image_moderation' in sys.modules:
    del sys.modules['services.image_moderation']

from services.text_moderation import TextModerationService
from services.image_moderation import ImageModerationService
from services.video_moderation import VideoModerationService
from services.audio_moderation import AudioModerationService
from dotenv import load_dotenv
load_dotenv()

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize services
text_service = TextModerationService()
image_service = ImageModerationService()
video_service = VideoModerationService()

# Initialize audio service dengan error handling
try:
    audio_service = AudioModerationService()
    print("[INFO] AudioModerationService initialized successfully")
except Exception as e:
    print(f"[ERROR] Failed to initialize AudioModerationService: {str(e)}")
    audio_service = None

# Policy settings (threshold defaults)
POLICY_SETTINGS = {
    'harassment_threshold': 0.5,
    'hate_threshold': 0.5,
    'violence_threshold': 0.5,
    'sexual_threshold': 0.5,
    'nudity_threshold': 0.5,
    'auto_approve_threshold': 0.3,
    'auto_reject_threshold': 0.7
}

# Mapping decision dari image_moderation.py ke format apply_policy
IMAGE_DECISION_MAP = {
    'DITOLAK': 'auto_reject',
    'PERLU_DITINJAU': 'review_required',
    'AMAN': 'auto_approve'
}

# File type yang diizinkan untuk audio upload
ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'm4a', 'ogg', 'flac', 'mp4', 'webm', 'wma', 'aac'}
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB

def allowed_audio_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Content Moderation API is running'
    })

# ============================================================================
# IMAGE PROXY ENDPOINTS - BYPASS CORS
# ============================================================================

@app.route('/proxy/image', methods=['GET'])
def proxy_image():
    """
    Proxy endpoint untuk fetch image dari URL external
    Usage: /proxy/image?url=https://example.com/image.jpg
    """
    try:
        image_url = request.args.get('url')
        
        if not image_url:
            return jsonify({'error': 'URL parameter required'}), 400
        
        if not (image_url.startswith('http://') or image_url.startswith('https://')):
            return jsonify({'error': 'Invalid URL'}), 400
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(image_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        content_type = response.headers.get('Content-Type', 'image/jpeg')
        
        return Response(
            response.content,
            mimetype=content_type,
            headers={
                'Access-Control-Allow-Origin': '*',
                'Cache-Control': 'public, max-age=3600'
            }
        )
        
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Failed to fetch image: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/proxy/extract-image', methods=['GET'])
def proxy_extract_image():
    """
    Extract image dari HTML page lalu proxy
    Usage: /proxy/extract-image?url=https://example.com/page
    """
    try:
        page_url = request.args.get('url')
        
        if not page_url:
            return jsonify({'error': 'URL parameter required'}), 400
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(page_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Try og:image meta tag first
        og_image = soup.find('meta', property='og:image')
        if og_image and og_image.get('content'):
            image_url = urljoin(page_url, og_image['content'])
        else:
            # Try first large image
            img_tag = soup.find('img')
            if img_tag and img_tag.get('src'):
                image_url = urljoin(page_url, img_tag['src'])
            else:
                return jsonify({'error': 'No image found in page'}), 404
        
        # Fetch extracted image
        img_response = requests.get(image_url, headers=headers, timeout=10)
        img_response.raise_for_status()
        
        content_type = img_response.headers.get('Content-Type', 'image/jpeg')
        
        return Response(
            img_response.content,
            mimetype=content_type,
            headers={
                'Access-Control-Allow-Origin': '*',
                'Cache-Control': 'public, max-age=3600',
                'X-Original-URL': image_url
            }
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# MODERATION ENDPOINTS
# ============================================================================

@app.route('/api/moderate/text', methods=['POST'])
def moderate_text():
    """
    Text moderation endpoint
    
    Body:
    {
        "text": "Text to moderate",
        "lang": "en" (optional, default: en)
    }
    """
    try:
        data = request.get_json()
        text = data.get('text')
        lang = data.get('lang', 'en')
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        result_obj = text_service.moderate(text, lang=lang)
        result = result_obj.to_dict()
        
        if result.get('decision') not in ['auto_approve', 'auto_reject', 'review_required']:
            decision = apply_policy(result)
            result['decision'] = decision
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/moderate/image', methods=['POST'])
def moderate_image():
    """
    Image moderation endpoint
    
    Body:
    {
        "image": "base64_string or URL",
        "type": "base64" or "url"
    }
    """
    try:
        data = request.get_json()
        image = data.get('image')
        image_type = data.get('type', 'base64')
        
        if not image:
            return jsonify({'error': 'Image is required'}), 400
        
        result_obj = image_service.moderate(image, image_type)
        result = result_obj.to_dict() if hasattr(result_obj, 'to_dict') else result_obj

        # FIX: image_moderation.py sudah return decision yang benar (DITOLAK / PERLU_DITINJAU / AMAN)
        # Map ke format policy dan jangan overwrite dengan apply_policy()
        raw_decision = result.get('decision')
        if raw_decision in IMAGE_DECISION_MAP:
            result['decision'] = IMAGE_DECISION_MAP[raw_decision]
        else:
            # Fallback ke apply_policy kalau decision nggak dikenal
            result['decision'] = apply_policy(result)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/moderate/video', methods=['POST'])
def moderate_video():
    """
    Video moderation endpoint
    
    Body:
    {
        "url": "Video URL"
    }
    """
    try:
        data = request.get_json()
        video_url = data.get('url')
        
        if not video_url:
            return jsonify({'error': 'Video URL is required'}), 400
        
        result_obj = video_service.moderate(video_url)
        result = result_obj.to_dict() if hasattr(result_obj, 'to_dict') else result_obj
        
        decision = apply_policy(result)
        result['decision'] = decision
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/moderate/audio', methods=['POST'])
def moderate_audio():
    """
    Audio moderation endpoint (URL mode)
    
    Body:
    {
        "audio_url": "Audio/Video URL",
        "language": "id" or "en" (optional, default: id)
    }
    """
    try:
        print("\n" + "="*60)
        print("[AUDIO MODERATION] Request received")
        print("="*60)
        
        # Cek service initialized
        if audio_service is None:
            print("[ERROR] Audio service not initialized")
            return jsonify({
                'status': 'error',
                'message': 'Audio service not initialized. Check ASSEMBLYAI_API_KEY in .env file'
            }), 500
        
        # Get request data
        data = request.get_json()
        print(f"[DEBUG] Request data: {data}")
        
        # Validasi
        if not data:
            print("[ERROR] No JSON data in request")
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided'
            }), 400
        
        # Validasi ketat: harus string dan non-empty
        audio_url = data.get('audio_url') or data.get('url')

        if not audio_url or not isinstance(audio_url, str):
            print(f"[ERROR] Invalid audio_url: {repr(audio_url)} (type: {type(audio_url).__name__})")
            return jsonify({
                'status': 'error',
                'message': 'audio_url is required and must be a valid URL string',
                'hint': 'Kirim body seperti: {"audio_url": "https://example.com/audio.mp3"}',
                'received': {
                    'value': repr(audio_url),
                    'type': type(audio_url).__name__
                }
            }), 400
        
        language = data.get('language', 'id')
        
        print(f"[INFO] Processing audio_url: {audio_url}")
        print(f"[INFO] Language: {language}")
        
        # Call audio moderation service
        result = audio_service.moderate(audio_url, language=language)
        
        print(f"[INFO] Moderation result status: {result.get('status')}")
        
        # Handle error result
        if result.get('status') == 'error':
            print(f"[ERROR] Moderation failed: {result.get('message')}")
            return jsonify(result), 500
        
        # Apply policy jika perlu
        if 'decision' not in result:
            decision = apply_policy(result)
            result['decision'] = decision
        
        print(f"[SUCCESS] Moderation completed. Flagged: {result.get('flagged')}")
        print("="*60 + "\n")
        
        return jsonify(result), 200
        
    except Exception as e:
        print(f"[EXCEPTION] Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Internal server error: {str(e)}'
        }), 500


@app.route('/api/moderate/audio/file', methods=['POST'])
def moderate_audio_file():
    """
    Audio moderation endpoint (file upload mode)
    
    Form data:
        audio: file upload
        language: "id" or "en" (optional, default: id)
    
    Flow:
        1. Terima file dari frontend
        2. Upload file ke AssemblyAI via /v2/upload
        3. Dapat upload_url dari AssemblyAI
        4. Kirim upload_url ke moderate() untuk transcribe + moderation
    """
    try:
        print("\n" + "="*60)
        print("[AUDIO MODERATION - FILE UPLOAD] Request received")
        print("="*60)

        if audio_service is None:
            return jsonify({
                'status': 'error',
                'message': 'Audio service not initialized. Check ASSEMBLYAI_API_KEY in .env file'
            }), 500

        # Cek file ada di request
        if 'audio' not in request.files:
            print("[ERROR] No 'audio' file in request")
            return jsonify({
                'status': 'error',
                'message': "File 'audio' tidak ditemukan dalam request"
            }), 400

        audio_file = request.files['audio']

        # Validasi filename
        if audio_file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'File kosong, pilih file audio yang valid'
            }), 400

        # Validasi extension
        if not allowed_audio_file(audio_file.filename):
            return jsonify({
                'status': 'error',
                'message': f'Format file tidak didukung. Gunakan: {", ".join(ALLOWED_AUDIO_EXTENSIONS)}'
            }), 400

        # Validasi size
        audio_file.seek(0, 2)  # Seek ke end
        file_size = audio_file.tell()
        audio_file.seek(0)     # Reset ke beginning

        if file_size > MAX_UPLOAD_SIZE:
            return jsonify({
                'status': 'error',
                'message': f'File terlalu besar. Maksimum 50MB, file Anda: {file_size / (1024*1024):.1f}MB'
            }), 400

        print(f"[INFO] File: {audio_file.filename}, Size: {file_size / 1024:.1f}KB")

        # Baca file bytes
        file_bytes = audio_file.read()

        # Step 1: Upload ke AssemblyAI
        print("[INFO] Uploading to AssemblyAI...")
        upload_url = audio_service.upload_audio(file_bytes)
        print(f"[INFO] AssemblyAI upload_url: {upload_url}")

        # Step 2: Moderate pake upload_url
        language = request.form.get('language', 'id')
        print(f"[INFO] Starting moderation with language: {language}")

        result = audio_service.moderate(upload_url, language=language)

        if result.get('status') == 'error':
            print(f"[ERROR] Moderation failed: {result.get('message')}")
            return jsonify(result), 500

        # Apply policy
        if 'decision' not in result:
            decision = apply_policy(result)
            result['decision'] = decision

        print(f"[SUCCESS] File moderation completed. Flagged: {result.get('flagged')}")
        print("="*60 + "\n")

        return jsonify(result), 200

    except Exception as e:
        print(f"[EXCEPTION] Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Internal server error: {str(e)}'
        }), 500


@app.route('/api/policy/settings', methods=['GET', 'POST'])
def policy_settings():
    """
    Get or update policy settings
    """
    global POLICY_SETTINGS
    
    if request.method == 'GET':
        return jsonify(POLICY_SETTINGS)
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            POLICY_SETTINGS.update(data)
            return jsonify({
                'status': 'success',
                'settings': POLICY_SETTINGS
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

def apply_policy(result):
    """
    Apply policy engine decision making
    
    Returns: 'auto_approve', 'auto_reject', or 'review_required'
    """
    if result.get('status') != 'success':
        return 'error'
    
    if not result.get('flagged'):
        return 'auto_approve'
    
    scores = result.get('scores', {})
    if scores:
        # FIX: filter cuma ambil value yang float/int
        # scores bisa mengandung dict seperti scores['nudity'] dan scores['weapon_classes']
        numeric_scores = [v for v in scores.values() if isinstance(v, (int, float))]

        # kalau scores['nudity'] ada (dict), ambil sub-scores-nya juga
        if 'nudity' in scores and isinstance(scores['nudity'], dict):
            numeric_scores.extend(scores['nudity'].values())

        # kalau scores['weapon_classes'] ada (dict), ambil sub-scores-nya juga
        if 'weapon_classes' in scores and isinstance(scores['weapon_classes'], dict):
            numeric_scores.extend(scores['weapon_classes'].values())

        if numeric_scores:
            max_score = max(numeric_scores)

            if max_score >= POLICY_SETTINGS['auto_reject_threshold']:
                return 'auto_reject'
            elif max_score <= POLICY_SETTINGS['auto_approve_threshold']:
                return 'auto_approve'
            else:
                return 'review_required'
    
    return 'review_required'

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get moderation statistics"""
    return jsonify({
        'total_requests': 0,
        'auto_approved': 0,
        'auto_rejected': 0,
        'pending_review': 0
    })

if __name__ == '__main__':
    # Cek environment variables
    if not os.getenv('ASSEMBLYAI_API_KEY'):
        print("\n" + "!"*60)
        print("WARNING: ASSEMBLYAI_API_KEY not set in .env file!")
        print("Audio moderation will not work without this API key")
        print("Get your free API key at: https://www.assemblyai.com/")
        print("!"*60 + "\n")
    
    if not os.getenv('SIGHTENGINE_API_USER') or not os.getenv('SIGHTENGINE_API_SECRET'):
        print("WARNING: Sightengine API credentials not set!")
        print("Please create .env file with SIGHTENGINE_API_USER and SIGHTENGINE_API_SECRET")
    
    print("=" * 60)
    print("CONTENT MODERATION API + IMAGE PROXY")
    print("=" * 60)
    print("Image Proxy Endpoints:")
    print("  GET /proxy/image?url=<URL>")
    print("  GET /proxy/extract-image?url=<URL>")
    print("Audio Endpoints:")
    print("  POST /api/moderate/audio          <- URL mode")
    print("  POST /api/moderate/audio/file     <- File upload mode")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)