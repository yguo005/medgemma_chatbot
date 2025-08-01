import json
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime

class ConversationState(Enum):
    INITIAL = "initial"
    SYMPTOM_DESCRIPTION = "symptom_description"
    DURATION_INQUIRY = "duration_inquiry"
    INTENSITY_INQUIRY = "intensity_inquiry"
    TIMING_INQUIRY = "timing_inquiry"
    DIAGNOSIS = "diagnosis"
    SERVICES = "services"
    COMPLETED = "completed"

class ConversationManager:
    def __init__(self):
        # In-memory storage for demo purposes
        # In production, use Redis or a database
        self.sessions: Dict[str, Dict] = {}
    
    def get_session(self, session_id: str) -> Dict:
        """Get or create a session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'state': ConversationState.INITIAL,
                'collected_data': {},
                'conversation_history': []
            }
        return self.sessions[session_id]
    
    def update_session(self, session_id: str, data: Dict):
        """Update session data"""
        session = self.get_session(session_id)
        session['collected_data'].update(data)
    
    def add_to_history(self, session_id: str, user_message: str, bot_response: str):
        """Add message to conversation history"""
        session = self.get_session(session_id)
        session['conversation_history'].append({
            'user': user_message,
            'bot': bot_response,
            'timestamp': str(datetime.now())
        })
    
    def process_message(self, session_id: str, message: str, is_choice: bool = False) -> Dict[str, Any]:
        """Process incoming message and return appropriate response"""
        session = self.get_session(session_id)
        current_state = session['state']
        
        if current_state == ConversationState.INITIAL:
            return self._handle_initial_state(session_id, message)
        elif current_state == ConversationState.SYMPTOM_DESCRIPTION:
            return self._handle_symptom_description(session_id, message)
        elif current_state == ConversationState.DURATION_INQUIRY:
            return self._handle_duration_inquiry(session_id, message, is_choice)
        elif current_state == ConversationState.INTENSITY_INQUIRY:
            return self._handle_intensity_inquiry(session_id, message, is_choice)
        elif current_state == ConversationState.TIMING_INQUIRY:
            return self._handle_timing_inquiry(session_id, message, is_choice)
        elif current_state == ConversationState.DIAGNOSIS:
            return self._handle_diagnosis(session_id, message)
        else:
            return {"response": "I'm not sure how to help with that. Let's start over."}
    
    def _handle_initial_state(self, session_id: str, message: str) -> Dict[str, Any]:
        """Handle initial symptom description"""
        session = self.get_session(session_id)
        session['collected_data']['symptoms'] = message
        session['state'] = ConversationState.DURATION_INQUIRY
        
        # Extract key symptom words to make duration question more specific
        symptom_type = self._extract_symptom_type(message)
        
        return {
            "response_type": "multiple_choice",
            "response_text": f"How long have you had this {symptom_type}?",
            "choices": [
                "Less than 3 days",
                "1 - 2 weeks", 
                "1 month",
                "More than 3 months",
                "More than 1 year"
            ]
        }
    
    def _handle_symptom_description(self, session_id: str, message: str) -> Dict[str, Any]:
        """Handle symptom description from photo/voice analysis"""
        return self._handle_initial_state(session_id, message)
    
    def _handle_duration_inquiry(self, session_id: str, message: str, is_choice: bool) -> Dict[str, Any]:
        """Handle duration selection"""
        session = self.get_session(session_id)
        session['collected_data']['duration'] = message
        session['state'] = ConversationState.INTENSITY_INQUIRY
        
        # Determine appropriate intensity question based on symptoms
        symptom_type = self._extract_symptom_type(session['collected_data'].get('symptoms', ''))
        
        if 'pain' in symptom_type.lower():
            intensity_question = "How would you describe the intensity of your pain?"
            choices = ["Mild", "Moderate", "Severe", "I'm not sure"]
        else:
            intensity_question = "How would you describe the severity of your symptoms?"
            choices = ["Mild", "Moderate", "Severe", "I'm not sure"]
        
        return {
            "response_type": "multiple_choice",
            "response_text": intensity_question,
            "choices": choices
        }
    
    def _handle_intensity_inquiry(self, session_id: str, message: str, is_choice: bool) -> Dict[str, Any]:
        """Handle intensity selection"""
        session = self.get_session(session_id)
        session['collected_data']['intensity'] = message
        session['state'] = ConversationState.TIMING_INQUIRY
        
        return {
            "response_type": "multiple_choice",
            "response_text": "Does it usually happen at a particular time?",
            "choices": [
                "In the morning",
                "Midday",
                "In the evening",
                "At nighttime",
                "None of the above"
            ]
        }
    
    def _handle_timing_inquiry(self, session_id: str, message: str, is_choice: bool) -> Dict[str, Any]:
        """Handle timing selection and provide diagnosis"""
        session = self.get_session(session_id)
        session['collected_data']['timing'] = message
        session['state'] = ConversationState.DIAGNOSIS
        
        # Generate diagnosis based on collected data
        diagnosis = self._generate_diagnosis(session['collected_data'])
        
        return {
            "response_type": "diagnostic",
            "diagnosis_title": diagnosis['title'],
            "diagnosis_description": diagnosis['description'],
            "recommendations": diagnosis['recommendations'],
            "next_action": "services"
        }
    
    def _handle_diagnosis(self, session_id: str, message: str) -> Dict[str, Any]:
        """Handle post-diagnosis interaction and show services"""
        session = self.get_session(session_id)
        session['state'] = ConversationState.SERVICES
        
        return {
            "response_type": "services",
            "services": [
                {
                    "id": "in_person_consultation",
                    "title": "In-Home Medical Consultation",
                    "price": "300",
                    "description": "On-demand home visit for medical diagnosis, treatment, and advice.",
                    "duration": "1 Hour"
                },
                {
                    "id": "virtual_consultation",
                    "title": "Virtual Medical Consultation", 
                    "price": "80",
                    "description": "Remote medical diagnosis, advice, and medication prescription.",
                    "duration": "45 Minutes"
                }
            ]
        }
    
    def _generate_diagnosis(self, collected_data: Dict) -> Dict[str, Any]:
        """Generate diagnosis based on collected symptoms"""
        symptoms = collected_data.get('symptoms', '').lower()
        duration = collected_data.get('duration', '')
        intensity = collected_data.get('intensity', '')
        timing = collected_data.get('timing', '')
        
        # More sophisticated symptom analysis
        if any(keyword in symptoms for keyword in ['knee', 'joint', 'leg']):
            if 'pain' in symptoms:
                return {
                    'title': 'Orthopedic Knee Condition',
                    'description': f'Based on your symptoms of {duration.lower()} {intensity.lower()} knee pain that occurs {timing.lower()}, this could indicate various knee conditions. Knee conditions often occur due to injury of the bones, ligaments, joint or soft tissue of the knee. It is commonly caused by physical trauma, overexertion, and inflammation.',
                    'recommendations': [
                        'Physical Therapist for mobility and swelling',
                        'Doctor for medical diagnosis', 
                        'Caregiver if you need home support or mobility help'
                    ]
                }
        elif any(keyword in symptoms for keyword in ['head', 'headache', 'migraine']):
            return {
                'title': 'Headache/Cephalgia Assessment',
                'description': f'You\'ve described {duration.lower()} {intensity.lower()} headaches occurring {timing.lower()}. Headaches can have various causes including tension, stress, dehydration, or underlying medical conditions.',
                'recommendations': [
                    'Doctor for proper diagnosis and treatment plan',
                    'Neurologist if headaches are severe or persistent'
                ]
            }
        elif any(keyword in symptoms for keyword in ['stomach', 'abdominal', 'belly', 'nausea']):
            return {
                'title': 'Gastrointestinal Symptoms',
                'description': f'Your {duration.lower()} {intensity.lower()} abdominal symptoms occurring {timing.lower()} could indicate various digestive issues that require medical evaluation.',
                'recommendations': [
                    'Doctor for medical diagnosis',
                    'Gastroenterologist if symptoms persist'
                ]
            }
        else:
            # Generic response for unrecognized symptoms
            return {
                'title': 'Medical Consultation Recommended',
                'description': f'Based on your {duration.lower()} {intensity.lower()} symptoms occurring {timing.lower()}, I recommend consulting with a healthcare professional for proper diagnosis and treatment. Your symptoms require medical evaluation to determine the appropriate care.',
                'recommendations': [
                    'Doctor for comprehensive medical diagnosis',
                    'Specialist consultation based on doctor\'s recommendation'
                ]
            }
    
    def _extract_symptom_type(self, symptoms: str) -> str:
        """Extract the main symptom type from description"""
        symptoms_lower = symptoms.lower()
        
        if 'pain' in symptoms_lower:
            if any(word in symptoms_lower for word in ['knee', 'leg', 'joint']):
                return 'knee pain'
            elif any(word in symptoms_lower for word in ['head', 'headache']):
                return 'headache'
            elif any(word in symptoms_lower for word in ['back']):
                return 'back pain'
            else:
                return 'pain'
        elif any(word in symptoms_lower for word in ['swelling', 'swollen']):
            return 'swelling'
        elif any(word in symptoms_lower for word in ['rash', 'skin']):
            return 'skin condition'
        else:
            return 'symptoms'
    
    def process_image_analysis(self, session_id: str, image_analysis: str) -> Dict[str, Any]:
        """Process image analysis result"""
        session = self.get_session(session_id)
        session['collected_data']['image_analysis'] = image_analysis
        session['state'] = ConversationState.DURATION_INQUIRY
        
        # Extract symptom type from image analysis
        symptom_type = self._extract_symptom_type(image_analysis)
        
        return {
            "response_type": "multiple_choice", 
            "response_text": f"Got it! Please help me understand more about the symptoms. {image_analysis}\n\nHow long have you had this {symptom_type}?",
            "choices": [
                "Less than 3 days",
                "1 - 2 weeks",
                "1 month", 
                "More than 3 months",
                "More than 1 year"
            ]
        }
    
    def process_voice_transcription(self, session_id: str, transcription: str) -> Dict[str, Any]:
        """Process voice transcription result"""
        return self.process_message(session_id, transcription)
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of the session for debugging or analysis"""
        session = self.get_session(session_id)
        return {
            'session_id': session_id,
            'current_state': session['state'].value if isinstance(session['state'], ConversationState) else str(session['state']),
            'collected_data': session['collected_data'],
            'conversation_length': len(session['conversation_history'])
        } 