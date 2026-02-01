import requests
import os
import time

class VideoModerationService:
    def __init__(self):
        self.api_user = os.getenv('SIGHTENGINE_API_USER')
        self.api_secret = os.getenv('SIGHTENGINE_API_SECRET')
        self.submit_url = 'https://api.sightengine.com/1.0/video/check-sync.json'
    
    def moderate(self, video_url):
        """
        Moderate video content
        
        Args:
            video_url: URL of the video to moderate
        
        Returns:
            dict: Moderation result
        """
        try:
            params = {
                'url': video_url,
                'models': 'nudity-2.1,gore,offensive,weapon,drugs,alcohol,hate,violence',
                'api_user': self.api_user,
                'api_secret': self.api_secret
            }
            
            response = requests.get(self.submit_url, params=params)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get('status') == 'success':
                # Parse frames
                frames = result.get('data', {}).get('frames', [])
                
                moderation_result = {
                    'status': 'success',
                    'flagged': False,
                    'total_frames': len(frames),
                    'flagged_frames': 0,
                    'categories': {},
                    'timeline': []
                }
                
                for frame in frames:
                    frame_flagged = False
                    frame_cats = []
                    
                    # Check nudity
                    if 'nudity' in frame and frame['nudity'].get('sexual_activity', 0) > 0.5:
                        frame_flagged = True
                        frame_cats.append('nudity')
                        moderation_result['categories']['nudity'] = True
                    
                    # Check gore
                    if 'gore' in frame and frame['gore'].get('prob', 0) > 0.5:
                        frame_flagged = True
                        frame_cats.append('gore')
                        moderation_result['categories']['gore'] = True
                    
                    # Check violence
                    if 'violence' in frame and frame['violence'].get('prob', 0) > 0.5:
                        frame_flagged = True
                        frame_cats.append('violence')
                        moderation_result['categories']['violence'] = True
                    
                    # Check weapon
                    if 'weapon' in frame and frame.get('weapon', 0) > 0.5:
                        frame_flagged = True
                        frame_cats.append('weapon')
                        moderation_result['categories']['weapon'] = True
                    
                    # Check hate
                    if 'hate' in frame and frame['hate'].get('prob', 0) > 0.5:
                        frame_flagged = True
                        frame_cats.append('hate')
                        moderation_result['categories']['hate'] = True
                    
                    if frame_flagged:
                        moderation_result['flagged'] = True
                        moderation_result['flagged_frames'] += 1
                        moderation_result['timeline'].append({
                            'time': frame.get('time', 0),
                            'categories': frame_cats
                        })
                
                return moderation_result
            else:
                return {
                    'status': 'error',
                    'message': result.get('error', {}).get('message', 'Unknown error')
                }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }