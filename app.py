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
audio_service = AudioModerationService()

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
    Audio moderation endpoint
    
    Body:
    {
        "url": "Audio/Video URL"
    }
    """
    try:
        data = request.get_json()
        audio_url = data.get('url')
        
        if not audio_url:
            return jsonify({'error': 'Audio URL is required'}), 400
        
        result_obj = audio_service.moderate(audio_url)
        result = result_obj.to_dict() if hasattr(result_obj, 'to_dict') else result_obj
        
        decision = apply_policy(result)
        result['decision'] = decision
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
    if not os.getenv('SIGHTENGINE_API_USER') or not os.getenv('SIGHTENGINE_API_SECRET'):
        print("WARNING: Sightengine API credentials not set!")
        print("Please create .env file with SIGHTENGINE_API_USER and SIGHTENGINE_API_SECRET")
    
    print("=" * 60)
    print("CONTENT MODERATION API + IMAGE PROXY")
    print("=" * 60)
    print("Image Proxy Endpoints:")
    print("  GET /proxy/image?url=<URL>")
    print("  GET /proxy/extract-image?url=<URL>")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)