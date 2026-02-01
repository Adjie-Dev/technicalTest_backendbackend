from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os

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

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Content Moderation API is running'
    })

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
        
        # FIX: Panggil .to_dict() untuk convert ke dictionary
        result_obj = text_service.moderate(text, lang=lang)
        result = result_obj.to_dict()
        
        # FIX: Jangan override decision jika sudah ada dari service
        # TextModerationService sudah return decision yang benar
        # apply_policy() hanya untuk fallback atau validasi tambahan
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
        
        # FIX: Panggil .to_dict() jika image_service juga return object
        result_obj = image_service.moderate(image, image_type)
        result = result_obj.to_dict() if hasattr(result_obj, 'to_dict') else result_obj
        
        # Apply policy
        decision = apply_policy(result)
        result['decision'] = decision
        
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
        
        # FIX: Panggil .to_dict() jika video_service juga return object
        result_obj = video_service.moderate(video_url)
        result = result_obj.to_dict() if hasattr(result_obj, 'to_dict') else result_obj
        
        # Apply policy
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
        
        # FIX: Panggil .to_dict() jika audio_service juga return object
        result_obj = audio_service.moderate(audio_url)
        result = result_obj.to_dict() if hasattr(result_obj, 'to_dict') else result_obj
        
        # Apply policy
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
    
    # Get highest score from all categories
    scores = result.get('scores', {})
    if scores:
        max_score = max(scores.values())
        
        if max_score >= POLICY_SETTINGS['auto_reject_threshold']:
            return 'auto_reject'
        elif max_score <= POLICY_SETTINGS['auto_approve_threshold']:
            return 'auto_approve'
        else:
            return 'review_required'
    
    # If flagged but no scores (rule-based only)
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
    # Check if API credentials are set
    if not os.getenv('SIGHTENGINE_API_USER') or not os.getenv('SIGHTENGINE_API_SECRET'):
        print("WARNING: Sightengine API credentials not set!")
        print("Please create .env file with SIGHTENGINE_API_USER and SIGHTENGINE_API_SECRET")
    
    app.run(debug=True, host='0.0.0.0', port=5000)