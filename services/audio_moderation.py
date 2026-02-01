import requests
import os

class AudioModerationService:
    def __init__(self):
        self.api_user = os.getenv('SIGHTENGINE_API_USER')
        self.api_secret = os.getenv('SIGHTENGINE_API_SECRET')
        self.base_url = 'https://api.sightengine.com/1.0/audio/check.json'
    
    def moderate(self, audio_url):
        """
        Moderate audio content (profanity in speech)
        
        Args:
            audio_url: URL of the audio/video to check for profanity
        
        Returns:
            dict: Moderation result
        """
        try:
            params = {
                'stream_url': audio_url,
                'models': 'speech',
                'api_user': self.api_user,
                'api_secret': self.api_secret
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            result = response.json()
            
            moderation_result = {
                'status': 'success',
                'flagged': False,
                'categories': {},
                'detections': []
            }
            
            # Check for profanity in speech
            if 'data' in result and 'profanity' in result['data']:
                profanity_data = result['data']['profanity']
                
                if profanity_data.get('matches'):
                    moderation_result['flagged'] = True
                    moderation_result['categories']['profanity'] = True
                    moderation_result['detections'] = profanity_data['matches']
            
            return moderation_result
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }