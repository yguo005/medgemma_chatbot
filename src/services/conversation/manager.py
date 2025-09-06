import json
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime, timedelta
import re
import asyncio
import logging

# Set up logging
logger = logging.getLogger(__name__)

class ConversationState(Enum):
    INITIAL = "initial"
    SYMPTOM_DESCRIPTION = "symptom_description"
    DURATION_INQUIRY = "duration_inquiry"
    INTENSITY_INQUIRY = "intensity_inquiry"
    TIMING_INQUIRY = "timing_inquiry"
    DIAGNOSIS = "diagnosis"
    SERVICES = "services"
    COMPLETED = "completed"
    FOLLOWUP = "followup"

class ConversationManager:
    def __init__(self, ai_service=None, rag_service=None):
        # In-memory storage for demo purposes
        # In production, use Redis or a database
        self.sessions: Dict[str, Dict] = {}
        self.session_timeout = timedelta(hours=1)  # Sessions expire after 1 hour
        
        # AI service for symptom understanding (Phase 1)
        self.ai_service = ai_service  # Will be MedGemmaService or ai_service_manager
        self.use_ai_extraction = ai_service is not None
        
        # RAG service for dynamic question generation (Phase 2)
        self.rag_service = rag_service  # Will be Chatbot (RAG system)
        self.use_rag_questions = rag_service is not None
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions to prevent memory leaks"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            last_activity = session.get('last_activity', current_time)
            if isinstance(last_activity, str):
                last_activity = datetime.fromisoformat(last_activity)
            
            if current_time - last_activity > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
    
    def get_session(self, session_id: str) -> Dict:
        """Get or create a session"""
        self.cleanup_expired_sessions()  # Periodic cleanup
        
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'state': ConversationState.INITIAL,
                'collected_data': {},
                'conversation_history': [],
                'created_at': datetime.now(),
                'last_activity': datetime.now(),
                'user_preferences': {}
            }
        else:
            # Update last activity
            self.sessions[session_id]['last_activity'] = datetime.now()
        
        return self.sessions[session_id]
    
    def update_session(self, session_id: str, data: Dict):
        """Update session data"""
        session = self.get_session(session_id)
        session['collected_data'].update(data)
        session['last_activity'] = datetime.now()
    
    def add_to_history(self, session_id: str, user_message: str, bot_response: str):
        """Add message to conversation history"""
        session = self.get_session(session_id)
        session['conversation_history'].append({
            'user': user_message,
            'bot': bot_response,
            'timestamp': datetime.now().isoformat(),
            'state': session['state'].value
        })
        session['last_activity'] = datetime.now()
    
    async def process_message(self, session_id: str, message: str, is_choice: bool = False) -> Dict[str, Any]:
        """Process incoming message and return appropriate response"""
        try:
            session = self.get_session(session_id)
            current_state = session['state']
            
            # Store the message for context
            session['collected_data']['last_message'] = message
            
            if current_state == ConversationState.INITIAL:
                return await self._handle_initial_state(session_id, message)
            elif current_state == ConversationState.SYMPTOM_DESCRIPTION:
                return await self._handle_symptom_description(session_id, message)
            elif current_state == ConversationState.DURATION_INQUIRY:
                return await self._handle_duration_inquiry(session_id, message, is_choice)
            elif current_state == ConversationState.INTENSITY_INQUIRY:
                return await self._handle_intensity_inquiry(session_id, message, is_choice)
            elif current_state == ConversationState.TIMING_INQUIRY:
                return await self._handle_timing_inquiry(session_id, message, is_choice)
            elif current_state == ConversationState.DIAGNOSIS:
                return self._handle_diagnosis(session_id, message)
            elif current_state == ConversationState.SERVICES:
                return self._handle_services_state(session_id, message)
            elif current_state == ConversationState.FOLLOWUP:
                return self._handle_followup(session_id, message)
            else:
                return self._handle_general_query(session_id, message)
        except Exception as e:
            logger.error(f" Error processing message for session {session_id}: {e}", exc_info=True)
            # Fallback response
            return {
                "response_type": "error",
                "response": "Sorry, something went wrong on our end. Please try again later.",
                "session_id": session_id
            }
    
    async def _handle_initial_state(self, session_id: str, message: str) -> Dict[str, Any]:
        """Handle initial symptom description with AI-powered entity extraction"""
        session = self.get_session(session_id)
        
        # Phase 1: AI for Symptom Understanding
        if self.use_ai_extraction:
            # Use AI to extract structured data from unstructured input
            try:
                structured_data = await self._extract_symptoms_with_ai(message)
                
                # Store both original message and structured data
                session['collected_data']['symptoms'] = message
                session['collected_data']['ai_extracted_data'] = structured_data
                
                # Use AI-extracted primary symptom for more accurate conversation flow
                symptom_type = structured_data.get('primary_symptom', 'symptoms')
                logger.info(f"AI extracted symptoms: {structured_data}")
                
                # Phase 2: Generate dynamic duration question using RAG
                if self.rag_service:
                    try:
                        dynamic_response = await self._generate_dynamic_question_with_rag(
                            primary_symptom=symptom_type,
                            collected_data=session['collected_data'],
                            question_type="duration"
                        )
                        
                        session['state'] = ConversationState.DURATION_INQUIRY
                        self.add_to_history(session_id, message, "Understanding your symptoms. Let me ask some follow-up questions.")
                        
                        logger.info(f" Generated dynamic duration question using RAG: {dynamic_response['response_text']}")
                        return dynamic_response
                        
                    except Exception as e:
                        logger.warning(f" RAG question generation failed, using fallback: {e}")
                
                # Fallback to AI-context duration question
                response_text = self._create_duration_question_with_ai_context(
                    symptom_type, structured_data.get('characteristics', []), structured_data.get('duration', '')
                )
                
            except Exception as e:
                logger.error(f"AI extraction failed, falling back to rule-based: {e}")
                # Fallback to original rule-based approach
                session['collected_data']['symptoms'] = message
                symptom_type = self._extract_symptom_type(message)
                response_text = f"How long have you been experiencing this {symptom_type}?"
        else:
            # Original rule-based approach
            session['collected_data']['symptoms'] = message
            symptom_type = self._extract_symptom_type(message)
            response_text = f"How long have you been experiencing this {symptom_type}?"
        
        session['state'] = ConversationState.DURATION_INQUIRY
        
        # Add message to history
        self.add_to_history(session_id, message, f"Understanding your symptoms. Let me ask some follow-up questions.")
        
        return {
            "response_type": "multiple_choice",
            "response_text": response_text,
            "choices": [
                "Less than 3 days",
                "1 - 2 weeks", 
                "1 month",
                "More than 3 months",
                "More than 1 year"
            ]
        }
    
    async def _handle_symptom_description(self, session_id: str, message: str) -> Dict[str, Any]:
        """Handle symptom description from photo/voice analysis"""
        return await self._handle_initial_state(session_id, message)
    
    async def _handle_duration_inquiry(self, session_id: str, message: str, is_choice: bool) -> Dict[str, Any]:
        """Handle duration selection with RAG-powered intensity question"""
        session = self.get_session(session_id)
        session['collected_data']['duration'] = message
        session['state'] = ConversationState.INTENSITY_INQUIRY
        
        # Get primary symptom for context
        primary_symptom = self._extract_primary_symptom(session['collected_data'])
        
        # Phase 2: Generate dynamic intensity question using RAG
        if self.rag_service:
            try:
                dynamic_response = await self._generate_dynamic_question_with_rag(
                    primary_symptom=primary_symptom,
                    collected_data=session['collected_data'],
                    question_type="intensity"
                )
                
                self.add_to_history(session_id, message, dynamic_response['response_text'])
                logger.info(f" Generated dynamic intensity question using RAG: {dynamic_response['response_text']}")
                return dynamic_response
                
            except Exception as e:
                logger.warning(f" Dynamic intensity question generation failed, using fallback: {e}")
        
        # Fallback: Use AI-extracted data for better context if available
        ai_data = session['collected_data'].get('ai_extracted_data', {})
        symptom_type = ai_data.get('primary_symptom', '') or self._extract_symptom_type(session['collected_data'].get('symptoms', ''))
        characteristics = ai_data.get('characteristics', [])
        
        # Create more intelligent intensity questions based on AI understanding
        if any(keyword in symptom_type.lower() for keyword in ['pain', 'ache', 'hurt']):
            if characteristics:
                intensity_question = f"You described {symptom_type} with {', '.join(characteristics[:2])} qualities. How would you rate the intensity?"
            else:
                intensity_question = "How would you describe the intensity of your pain?"
            choices = ["Mild (1-3)", "Moderate (4-6)", "Severe (7-10)", "I'm not sure"]
        elif any(keyword in symptom_type.lower() for keyword in ['swelling', 'rash', 'inflammation']):
            intensity_question = "How would you describe the severity of the affected area?"
            choices = ["Mild", "Moderate", "Severe", "I'm not sure"]
        else:
            intensity_question = "How would you describe the severity of your symptoms?"
            choices = ["Mild", "Moderate", "Severe", "I'm not sure"]
        
        self.add_to_history(session_id, message, intensity_question)
        
        return {
            "response_type": "multiple_choice",
            "response_text": intensity_question,
            "choices": choices
        }
    
    async def _handle_intensity_inquiry(self, session_id: str, message: str, is_choice: bool) -> Dict[str, Any]:
        """Handle intensity selection with RAG-powered timing question"""
        session = self.get_session(session_id)
        session['collected_data']['intensity'] = message
        session['state'] = ConversationState.TIMING_INQUIRY
        
        # Get primary symptom for context
        primary_symptom = self._extract_primary_symptom(session['collected_data'])
        
        # Phase 2: Generate dynamic timing question using RAG
        if self.rag_service:
            try:
                dynamic_response = await self._generate_dynamic_question_with_rag(
                    primary_symptom=primary_symptom,
                    collected_data=session['collected_data'],
                    question_type="timing"
                )
                
                self.add_to_history(session_id, message, dynamic_response['response_text'])
                logger.info(f" Generated dynamic timing question using RAG: {dynamic_response['response_text']}")
                return dynamic_response
                
            except Exception as e:
                logger.warning(f" Dynamic timing question generation failed, using fallback: {e}")
        
        # Fallback timing question
        timing_question = "When do these symptoms typically occur or get worse?"
        choices = [
            "In the morning",
            "During the day",
            "In the evening",
            "At night",
            "After physical activity",
            "No specific pattern"
        ]
        
        self.add_to_history(session_id, message, timing_question)
        
        return {
            "response_type": "multiple_choice", 
            "response_text": timing_question,
            "choices": choices[:4]  # Limit to 4 choices for consistency
        }
    
    async def _handle_timing_inquiry(self, session_id: str, message: str, is_choice: bool) -> Dict[str, Any]:
        """Handle timing selection and provide diagnosis with Phase 3 final explanation"""
        session = self.get_session(session_id)
        session['collected_data']['timing'] = message
        session['state'] = ConversationState.DIAGNOSIS
        
        # Generate diagnosis based on collected data
        diagnosis = self._generate_diagnosis(session['collected_data'])
        
        # Phase 3: Generate final explanation using RAG + AI
        final_explanation_result = {"explanation": "", "success": False}
        if self.rag_service:
            try:
                final_explanation_result = await self._generate_final_explanation_with_rag(diagnosis['title'])
                logger.info(f" Generated Phase 3 final explanation: {final_explanation_result['explanation'][:100]}...")
            except Exception as e:
                logger.warning(f" Phase 3 explanation generation failed: {e}")
                final_explanation_result = self._generate_fallback_explanation(diagnosis['title'])
        else:
            logger.info(" RAG service not available, using fallback explanation")
            final_explanation_result = self._generate_fallback_explanation(diagnosis['title'])
        
        self.add_to_history(session_id, message, f"Diagnosis: {diagnosis['title']}")
        
        return {
            "response_type": "diagnostic",
            "diagnosis_title": diagnosis['title'],
            "diagnosis_description": diagnosis['description'],
            "final_explanation": final_explanation_result['explanation'],  # Phase 3 result
            "explanation_source": final_explanation_result.get('generation_method', 'unknown'),
            "key_terms_explained": final_explanation_result.get('key_terms', []),
            "recommendations": diagnosis['recommendations'],
            "urgency_level": diagnosis.get('urgency_level', 'moderate'),
            "next_action": "services"
        }
    
    def _handle_diagnosis(self, session_id: str, message: str) -> Dict[str, Any]:
        """Handle post-diagnosis interaction and show services"""
        session = self.get_session(session_id)
        session['state'] = ConversationState.SERVICES
        
        # Check if user has questions about the diagnosis
        question_keywords = ['what', 'how', 'why', 'explain', 'tell me more', 'details']
        if any(keyword in message.lower() for keyword in question_keywords):
            return {
                "response_type": "text",
                "response": "I'd be happy to explain more. Based on your symptoms, I've provided a preliminary assessment. However, for a complete diagnosis and treatment plan, I recommend consulting with one of our healthcare professionals. Here are the available services:"
            }
        
        return self._show_services(session_id)
    
    def _handle_services_state(self, session_id: str, message: str) -> Dict[str, Any]:
        """Handle interactions in services state"""
        session = self.get_session(session_id)
        
        # Check if user wants to book or ask about services
        if any(keyword in message.lower() for keyword in ['book', 'schedule', 'appointment', 'consultation']):
            session['state'] = ConversationState.COMPLETED
            return {
                "response_type": "text",
                "response": "Great! I'll help you book an appointment. For demo purposes, this would connect to a real booking system. Thank you for using AI Health Consultant!"
            }
        elif any(keyword in message.lower() for keyword in ['price', 'cost', 'insurance', 'payment']):
            return {
                "response_type": "text",
                "response": "Our pricing is transparent and competitive. Virtual consultations are $80 for 45 minutes, and in-home visits are $300 for 1 hour. We accept most major insurance plans. Would you like to proceed with booking?"
            }
        else:
            session['state'] = ConversationState.FOLLOWUP
            return {
                "response_type": "text",
                "response": "Is there anything else about your symptoms or our services that you'd like to know? I'm here to help!"
            }
    
    def _handle_followup(self, session_id: str, message: str) -> Dict[str, Any]:
        """Handle follow-up questions and general queries"""
        return self._handle_general_query(session_id, message)
    
    def _handle_general_query(self, session_id: str, message: str) -> Dict[str, Any]:
        """Handle general queries that don't fit the main flow"""
        session = self.get_session(session_id)
        
        # Check for common queries
        if any(keyword in message.lower() for keyword in ['start over', 'restart', 'new symptoms']):
            session['state'] = ConversationState.INITIAL
            session['collected_data'] = {}
            return {
                "response_type": "text",
                "response": "Of course! Let's start fresh. Please describe your current symptoms or concerns."
            }
        elif any(keyword in message.lower() for keyword in ['emergency', 'urgent', 'severe', 'call 911']):
            return {
                "response_type": "text",
                "response": " If this is a medical emergency, please call 911 or go to your nearest emergency room immediately. For urgent but non-emergency concerns, consider visiting an urgent care center or contacting your primary care physician."
            }
        else:
            return {
                "response_type": "text",
                "response": "I understand you have questions. For the best medical guidance, I recommend booking a consultation with one of our healthcare professionals. They can provide personalized advice for your specific situation."
            }
    
    def _show_services(self, session_id: str) -> Dict[str, Any]:
        """Show available services based on collected data"""
        session = self.get_session(session_id)
        collected_data = session['collected_data']
        
        # Customize services based on symptoms and urgency
        urgency = self._assess_urgency(collected_data)
        
        if urgency == 'high':
            services = [
                {
                    "id": "urgent_consultation",
                    "title": "Urgent Virtual Consultation",
                    "price": "120",
                    "description": "Same-day virtual medical consultation for urgent symptoms.",
                    "duration": "30 Minutes",
                    "availability": "Available today"
                },
                {
                    "id": "in_person_urgent",
                    "title": "Urgent In-Home Visit",
                    "price": "400",
                    "description": "Same-day in-home medical visit for urgent care.",
                    "duration": "45 Minutes",
                    "availability": "Available within 4 hours"
                }
            ]
        else:
            services = [
                {
                    "id": "virtual_consultation",
                    "title": "Virtual Medical Consultation", 
                    "price": "80",
                    "description": "Remote medical diagnosis, advice, and medication prescription.",
                    "duration": "45 Minutes",
                    "availability": "Next available: Tomorrow"
                },
                {
                    "id": "in_person_consultation",
                    "title": "In-Home Medical Consultation",
                    "price": "300",
                    "description": "On-demand home visit for medical diagnosis, treatment, and advice.",
                    "duration": "1 Hour",
                    "availability": "Next available: Within 2 days"
                }
            ]
        
        return {
            "response_type": "services",
            "services": services,
            "urgency_level": urgency
        }
    
    def _generate_diagnosis(self, collected_data: Dict) -> Dict[str, Any]:
        """Generate diagnosis based on collected symptoms with AI-enhanced logic"""
        symptoms = collected_data.get('symptoms', '').lower()
        duration = collected_data.get('duration', '')
        intensity = collected_data.get('intensity', '')
        timing = collected_data.get('timing', '')
        
        # Use AI-extracted data for better diagnosis if available
        ai_data = collected_data.get('ai_extracted_data', {})
        primary_symptom = ai_data.get('primary_symptom', '').lower()
        characteristics = ai_data.get('characteristics', [])
        severity_indicators = ai_data.get('severity_indicators', [])
        location = ai_data.get('location', '')
        
        # Combine AI and traditional logic for better diagnosis
        symptom_context = primary_symptom or symptoms
        
        # Initialize diagnosis components
        urgency_level = 'moderate'
        
        # Enhanced symptom analysis using AI-extracted data
        if any(keyword in symptom_context for keyword in ['knee', 'joint', 'leg']):
            if any(keyword in symptom_context for keyword in ['pain', 'hurt', 'ache']) or 'pain' in primary_symptom:
                if 'severe' in intensity.lower() or any(sev in severity_indicators for sev in ['severe', 'unbearable']):
                    urgency_level = 'high'
                
                # Enhanced description using AI characteristics
                char_desc = f" with {', '.join(characteristics)}" if characteristics else ""
                location_desc = f" in the {location}" if location else ""
                    
                return {
                    'title': 'Orthopedic Knee Assessment',
                    'description': f'Based on your {duration.lower()} {intensity.lower()} knee symptoms{char_desc}{location_desc} that occur {timing.lower()}, this could indicate various knee conditions ranging from minor strain to more significant joint issues. Knee problems commonly result from injury to bones, ligaments, cartilage, or soft tissue, often caused by physical activity, overuse, or trauma.',
                    'recommendations': [
                        'Orthopedic specialist for comprehensive joint evaluation',
                        'Physical therapist for mobility assessment and treatment',
                        'Consider imaging studies (X-ray or MRI) if symptoms persist',
                        'Pain management specialist if conservative treatment fails'
                    ],
                    'urgency_level': urgency_level
                }
                
        elif any(keyword in symptom_context for keyword in ['head', 'headache', 'migraine']) or 'headache' in primary_symptom:
            if any(keyword in symptom_context for keyword in ['severe', 'worst', 'sudden']) or any(sev in severity_indicators for sev in ['severe', 'worst']):
                urgency_level = 'high'
            
            # Enhanced description using AI characteristics
            char_desc = f" described as {', '.join(characteristics)}" if characteristics else ""
            location_desc = f" {location}" if location else ""
                
            return {
                'title': 'Headache/Cephalgia Assessment',
                'description': f'You\'ve described {duration.lower()} {intensity.lower()} headaches{char_desc}{location_desc} occurring {timing.lower()}. Headaches can have various causes including tension, stress, dehydration, medication overuse, or underlying medical conditions. The pattern and characteristics help determine the most appropriate treatment approach.',
                'recommendations': [
                    'Primary care physician for initial evaluation and treatment plan',
                    'Neurologist consultation if headaches are frequent, severe, or changing',
                    'Keep a headache diary to track triggers and patterns',
                    'Consider lifestyle modifications (sleep, stress, hydration)'
                ],
                'urgency_level': urgency_level
            }
            
        elif any(keyword in symptom_context for keyword in ['stomach', 'abdominal', 'belly', 'nausea', 'vomit']):
            if any(keyword in symptom_context for keyword in ['severe', 'blood', 'unable to keep']) or any(sev in severity_indicators for sev in ['severe']):
                urgency_level = 'high'
                
            return {
                'title': 'Gastrointestinal Symptoms Assessment',
                'description': f'Your {duration.lower()} {intensity.lower()} abdominal symptoms occurring {timing.lower()} could indicate various digestive issues. These may range from dietary intolerances to inflammatory conditions or infections that require medical evaluation for proper diagnosis and treatment.',
                'recommendations': [
                    'Primary care physician or gastroenterologist for evaluation',
                    'Consider dietary modifications and symptom tracking',
                    'Hydration and electrolyte management if experiencing nausea/vomiting',
                    'Specialist referral if symptoms persist or worsen'
                ],
                'urgency_level': urgency_level
            }
            
        elif any(keyword in symptom_context for keyword in ['chest', 'heart', 'breathing', 'shortness']):
            urgency_level = 'high'  # Chest symptoms often require urgent evaluation
            
            return {
                'title': 'Cardiopulmonary Symptoms - Requires Evaluation',
                'description': f'Chest-related symptoms like yours require prompt medical attention to rule out serious conditions. While many chest symptoms are benign, it\'s important to have a healthcare professional evaluate breathing difficulties, chest pain, or heart-related concerns.',
                'recommendations': [
                    'Seek immediate medical attention if symptoms are severe',
                    'Emergency room visit if experiencing severe chest pain or breathing difficulty',
                    'Cardiology consultation for heart-related concerns',
                    'Pulmonology referral for persistent breathing issues'
                ],
                'urgency_level': urgency_level
            }
            
        else:
            # Generic response for unrecognized symptoms
            ai_context = f" with {', '.join(characteristics[:2])}" if characteristics else ""
            return {
                'title': 'Medical Consultation Recommended',
                'description': f'Based on your {duration.lower()} {intensity.lower()} symptoms{ai_context} that occur {timing.lower()}, a comprehensive medical evaluation would be beneficial. While I can provide general guidance, your symptoms deserve professional medical attention to determine the appropriate diagnosis and treatment plan.',
                'recommendations': [
                    'Primary care physician for comprehensive medical evaluation',
                    'Specialist consultation based on primary care recommendations',
                    'Symptom tracking and documentation for your appointment',
                    'Consider urgent care if symptoms worsen or new symptoms develop'
                ],
                'urgency_level': urgency_level
            }
    
    async def _extract_symptoms_with_ai(self, user_input: str) -> Dict[str, Any]:
        """
        Phase 1: AI for Symptom Understanding
        Extract structured medical data from unstructured user input using AI
        """
        # Construct the AI prompt for entity extraction
        extraction_prompt = f"""Analyze the following medical symptom description and extract structured information in JSON format.

User Input: "{user_input}"

Extract the following information:
1. primary_symptom: The main medical complaint (e.g., "headache", "knee pain", "fever")
2. characteristics: List of descriptive qualities (e.g., ["pounding", "behind eyes", "throbbing"])
3. duration: Any mentioned time period (e.g., "two days", "3 weeks", "since yesterday")
4. associated_symptoms: Other symptoms mentioned (e.g., ["dizziness", "nausea"])
5. location: Body part or area affected (e.g., "behind eyes", "right knee", "chest")
6. severity_indicators: Words indicating intensity (e.g., ["severe", "mild", "unbearable"])

Return ONLY a valid JSON object with these fields. If information is not mentioned, use empty string or empty list.

Example format:
{{
  "primary_symptom": "headache",
  "characteristics": ["pounding", "behind eyes"],
  "duration": "two days",
  "associated_symptoms": ["dizziness when standing"],
  "location": "behind eyes",
  "severity_indicators": ["severe"]
}}"""

        try:
            # Call AI service for extraction
            if hasattr(self.ai_service, 'generate_medical_response'):
                # Use MedGemmaService directly
                ai_response = await self.ai_service.generate_medical_response(
                    query=extraction_prompt,
                    max_length=300,
                    temperature=0.1  # Low temperature for consistent extraction
                )
                
                if ai_response.get('success'):
                    response_text = ai_response.get('response', '')
                else:
                    raise Exception(f"AI service error: {ai_response.get('error', 'Unknown error')}")
                    
            elif hasattr(self.ai_service, 'analyze_symptoms_text'):
                # Use ai_service_manager or similar service
                ai_response = await self.ai_service.analyze_symptoms_text(
                    symptoms=extraction_prompt
                )
                
                if ai_response.get('success'):
                    response_text = ai_response.get('response', '')
                else:
                    raise Exception(f"AI service error: {ai_response.get('error', 'Unknown error')}")
            else:
                raise Exception("AI service does not have required methods")
            
            # Parse the JSON response
            structured_data = self._parse_ai_extraction_response(response_text)
            
            logger.info(f" AI extraction successful: {structured_data}")
            return structured_data
            
        except Exception as e:
            logger.error(f" AI extraction failed: {e}")
            # Return fallback structured data
            return self._fallback_symptom_extraction(user_input)
    
    def _parse_ai_extraction_response(self, ai_response: str) -> Dict[str, Any]:
        """Parse the AI's JSON response and validate structure"""
        try:
            # Try to find JSON in the response
            import re
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed_data = json.loads(json_str)
                
                # Validate required fields
                required_fields = ['primary_symptom', 'characteristics', 'duration', 
                                 'associated_symptoms', 'location', 'severity_indicators']
                
                validated_data = {}
                for field in required_fields:
                    validated_data[field] = parsed_data.get(field, [] if field in ['characteristics', 'associated_symptoms', 'severity_indicators'] else '')
                
                return validated_data
            else:
                raise ValueError("No JSON found in AI response")
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse AI JSON response: {e}")
            raise Exception("Invalid JSON response from AI")
    
    def _fallback_symptom_extraction(self, user_input: str) -> Dict[str, Any]:
        """Fallback extraction using rule-based approach when AI fails"""
        logger.info("Using fallback rule-based extraction")
        
        # Use existing rule-based logic as fallback
        primary_symptom = self._extract_symptom_type(user_input)
        
        # Simple rule-based extraction
        user_lower = user_input.lower()
        
        # Extract characteristics using keywords
        characteristics = []
        char_keywords = ['pounding', 'sharp', 'dull', 'throbbing', 'burning', 'aching', 
                        'stabbing', 'cramping', 'shooting', 'tingling']
        for keyword in char_keywords:
            if keyword in user_lower:
                characteristics.append(keyword)
        
        # Extract duration hints
        duration = ""
        duration_patterns = [
            r'(\d+)\s*(day|days|week|weeks|month|months|year|years)',
            r'(yesterday|today|last night|this morning)',
            r'(since\s+\w+)'
        ]
        for pattern in duration_patterns:
            match = re.search(pattern, user_lower)
            if match:
                duration = match.group()
                break
        
        return {
            "primary_symptom": primary_symptom,
            "characteristics": characteristics,
            "duration": duration,
            "associated_symptoms": [],
            "location": "",
            "severity_indicators": []
        }
    
    async def _generate_dynamic_question_with_rag(self, 
                                                primary_symptom: str, 
                                                collected_data: Dict[str, Any], 
                                                question_type: str) -> Dict[str, Any]:
        """
        Phase 2: AI for Dynamic Question Generation using RAG
        Generate clinically relevant follow-up questions using medical encyclopedia context
        """
        try:
            # Step 1: Retrieve relevant clinical context from medical encyclopedia
            rag_query = f"{primary_symptom} assessment {question_type} clinical evaluation"
            
            logger.info(f" Retrieving RAG context for: {rag_query}")
            
            # Query the RAG system for relevant medical context
            clinical_context = await self.rag_service.get_response(query=rag_query)
            
            logger.info(f" Retrieved clinical context: {clinical_context[:200]}...")
            
            # Step 2: Construct user data context
            user_context = self._construct_user_data_context(collected_data)
            
            # Step 3: Generate dynamic question using AI + RAG
            question_prompt = f"""You are a medical AI assistant generating the next most important follow-up question for a patient assessment.

CLINICAL CONTEXT FROM MEDICAL ENCYCLOPEDIA:
{clinical_context}

CURRENT PATIENT DATA:
{user_context}

TASK: Based on the clinical context above and the patient's current data, what is the single most important {question_type} question to ask next?

REQUIREMENTS:
1. The question must be clinically relevant based on the medical encyclopedia context
2. Consider what information would be most valuable for assessment
3. Provide exactly 4 multiple-choice options
4. Make the question specific to the patient's symptoms
5. Use proper medical terminology but keep it patient-friendly

OUTPUT FORMAT (JSON only):
{{
  "question": "Your specific question here",
  "choices": ["Option 1", "Option 2", "Option 3", "Option 4"],
  "clinical_rationale": "Brief explanation of why this question is important"
}}

Return ONLY the JSON object, no additional text."""

            # Call AI service to generate the question
            if hasattr(self.ai_service, 'generate_medical_response'):
                ai_response = await self.ai_service.generate_medical_response(
                    query=question_prompt,
                    max_length=400,
                    temperature=0.3  # Moderate creativity for question generation
                )
                
                if ai_response.get('success'):
                    response_text = ai_response.get('response', '')
                else:
                    raise Exception(f"AI service error: {ai_response.get('error', 'Unknown error')}")
            else:
                raise Exception("AI service does not support question generation")
            
            # Parse the AI-generated question
            dynamic_question = self._parse_dynamic_question_response(response_text)
            
            logger.info(f" Generated dynamic question: {dynamic_question['question']}")
            
            return {
                "response_type": "multiple_choice",
                "response_text": dynamic_question['question'],
                "choices": dynamic_question['choices'],
                "clinical_rationale": dynamic_question.get('clinical_rationale', ''),
                "generation_method": "rag_dynamic"
            }
            
        except Exception as e:
            logger.error(f" Dynamic question generation failed: {e}")
            # Fallback to static questions
            return self._generate_fallback_question(primary_symptom, question_type)
    
    def _construct_user_data_context(self, collected_data: Dict[str, Any]) -> str:
        """Construct a summary of user data for AI context"""
        context_parts = []
        
        # Original symptoms
        if collected_data.get('symptoms'):
            context_parts.append(f"Original complaint: {collected_data['symptoms']}")
        
        # AI-extracted data
        ai_data = collected_data.get('ai_extracted_data', {})
        if ai_data:
            if ai_data.get('primary_symptom'):
                context_parts.append(f"Primary symptom: {ai_data['primary_symptom']}")
            if ai_data.get('characteristics'):
                context_parts.append(f"Characteristics: {', '.join(ai_data['characteristics'])}")
            if ai_data.get('location'):
                context_parts.append(f"Location: {ai_data['location']}")
            if ai_data.get('duration'):
                context_parts.append(f"Duration mentioned: {ai_data['duration']}")
            if ai_data.get('associated_symptoms'):
                context_parts.append(f"Associated symptoms: {', '.join(ai_data['associated_symptoms'])}")
        
        # Previously collected answers
        if collected_data.get('duration'):
            context_parts.append(f"Duration selected: {collected_data['duration']}")
        if collected_data.get('intensity'):
            context_parts.append(f"Intensity: {collected_data['intensity']}")
        if collected_data.get('timing'):
            context_parts.append(f"Timing pattern: {collected_data['timing']}")
        
        return "\n".join(context_parts) if context_parts else "No specific data collected yet"
    
    def _parse_dynamic_question_response(self, ai_response: str) -> Dict[str, Any]:
        """Parse the AI's dynamic question JSON response"""
        try:
            # Try to find JSON in the response
            import re
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed_data = json.loads(json_str)
                
                # Validate required fields
                if 'question' not in parsed_data or 'choices' not in parsed_data:
                    raise ValueError("Missing required fields in AI response")
                
                if not isinstance(parsed_data['choices'], list) or len(parsed_data['choices']) != 4:
                    raise ValueError("Choices must be a list of exactly 4 options")
                
                return {
                    'question': parsed_data['question'],
                    'choices': parsed_data['choices'],
                    'clinical_rationale': parsed_data.get('clinical_rationale', '')
                }
            else:
                raise ValueError("No JSON found in AI response")
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse dynamic question JSON: {e}")
            raise Exception("Invalid JSON response from AI for question generation")
    
    def _generate_fallback_question(self, primary_symptom: str, question_type: str) -> Dict[str, Any]:
        """Generate fallback question when RAG+AI fails"""
        logger.info(f"Using fallback question generation for {primary_symptom} - {question_type}")
        
        fallback_questions = {
            "duration": {
                "question": f"How long have you been experiencing this {primary_symptom}?",
                "choices": ["Less than 3 days", "1 - 2 weeks", "1 month", "More than 3 months", "More than 1 year"]
            },
            "intensity": {
                "question": f"How would you describe the severity of your {primary_symptom}?",
                "choices": ["Mild (1-3)", "Moderate (4-6)", "Severe (7-10)", "I'm not sure"]
            },
            "timing": {
                "question": "When do these symptoms typically occur or get worse?",
                "choices": ["In the morning", "During the day", "In the evening", "At night", "After physical activity", "No specific pattern"]
            }
        }
        
        fallback = fallback_questions.get(question_type, fallback_questions["duration"])
        
        return {
            "response_type": "multiple_choice",
            "response_text": fallback["question"],
            "choices": fallback["choices"][:4],  # Ensure exactly 4 choices
            "generation_method": "fallback"
        }

    async def _generate_final_explanation_with_rag(self, diagnosis_title: str) -> Dict[str, Any]:
        """
        Phase 3: AI for Generating the Final Explanation using RAG
        Generate trustworthy, user-friendly explanations based on medical encyclopedia context
        """
        try:
            # Step 1: Extract key medical terms from diagnosis title
            key_terms = self._extract_medical_terms_from_diagnosis(diagnosis_title)
            logger.info(f" Extracted key terms for explanation: {key_terms}")
            
            # Step 2: Retrieve factual context from medical encyclopedia using RAG
            rag_query = f"{' '.join(key_terms)} definition medical encyclopedia explanation"
            
            if self.rag_service:
                encyclopedia_context = await self.rag_service.get_response(query=rag_query)
                
                logger.info(f" Retrieved encyclopedia context: {encyclopedia_context[:200]}...")
            else:
                encyclopedia_context = "No trusted medical context available."
            
            # Step 3: Generate user-friendly explanation using AI + RAG context
            explanation_prompt = f"""You are a medical AI assistant creating a patient-friendly explanation based on trusted medical sources.

CONTEXT FROM MEDICAL ENCYCLOPEDIA:
{encyclopedia_context}

DIAGNOSIS TITLE: {diagnosis_title}

TASK: Based on the medical encyclopedia context above, write a simple, easy-to-understand explanation for a patient about their condition.

REQUIREMENTS:
1. Use the factual information from the encyclopedia context
2. Explain medical terms in simple language
3. Focus on what the condition is and why medical consultation is important
4. Keep the tone reassuring but emphasize professional medical guidance
5. Do not diagnose or provide specific medical advice
6. Limit to 2-3 sentences

OUTPUT FORMAT: Return only the plain text explanation, no additional formatting."""

            # Call AI service to generate the explanation
            if hasattr(self.ai_service, 'generate_medical_response'):
                ai_response = await self.ai_service.generate_medical_response(
                    query=explanation_prompt,
                    max_length=300,
                    temperature=0.2  # Low temperature for consistent, factual responses
                )
                
                if ai_response.get('success'):
                    explanation = ai_response.get('response', '').strip()
                else:
                    raise Exception(f"AI service error: {ai_response.get('error', 'Unknown error')}")
            else:
                raise Exception("AI service does not support explanation generation")
            
            logger.info(f" Generated final explanation: {explanation[:100]}...")
            
            return {
                "explanation": explanation,
                "encyclopedia_context": encyclopedia_context[:500],  # Include source context
                "key_terms": key_terms,
                "generation_method": "rag_explanation",
                "success": True
            }
            
        except Exception as e:
            logger.error(f" Final explanation generation failed: {e}")
            return self._generate_fallback_explanation(diagnosis_title)
    
    def _extract_medical_terms_from_diagnosis(self, diagnosis_title: str) -> List[str]:
        """Extract key medical terms from diagnosis title for RAG query"""
        # Common medical terms mapping
        medical_term_patterns = {
            'headache': ['headache', 'cephalgia'],
            'cephalgia': ['headache', 'cephalgia'], 
            'migraine': ['migraine', 'headache'],
            'knee': ['knee', 'joint'],
            'pain': ['pain', 'ache'],
            'orthopedic': ['orthopedic', 'bone', 'joint'],
            'gastrointestinal': ['gastrointestinal', 'digestive', 'stomach'],
            'cardiopulmonary': ['cardiopulmonary', 'heart', 'lung'],
            'respiratory': ['respiratory', 'breathing', 'lung'],
            'dermatological': ['dermatological', 'skin'],
            'neurological': ['neurological', 'nervous system']
        }
        
        diagnosis_lower = diagnosis_title.lower()
        key_terms = []
        
        # Extract terms based on patterns
        for term, related_terms in medical_term_patterns.items():
            if term in diagnosis_lower:
                key_terms.extend(related_terms)
        
        # Fallback: extract first few words if no specific patterns found
        if not key_terms:
            words = diagnosis_title.split()[:2]  # Take first 2 words
            key_terms = [word.lower().strip('/:') for word in words if len(word) > 3]
        
        return list(set(key_terms))  # Remove duplicates
    
    def _generate_fallback_explanation(self, diagnosis_title: str) -> Dict[str, Any]:
        """Generate fallback explanation when RAG+AI fails"""
        logger.info(f"Using fallback explanation for: {diagnosis_title}")
        
        fallback_explanations = {
            "headache": "Headaches are a common condition involving pain in the head or neck area. They can have various causes and it's important to consult with a healthcare professional for proper evaluation and treatment.",
            "knee": "Knee problems can involve the bones, joints, or soft tissues of the knee area. A healthcare professional can help determine the cause and recommend appropriate treatment options.",
            "chest": "Chest-related symptoms require prompt medical attention to ensure proper evaluation. A healthcare professional can assess your condition and provide appropriate guidance.",
            "gastrointestinal": "Digestive symptoms can have various causes and may require medical evaluation. It's important to consult with a healthcare professional for proper assessment and treatment recommendations."
        }
        
        # Find appropriate fallback based on diagnosis content
        diagnosis_lower = diagnosis_title.lower()
        explanation = "This condition may require medical evaluation to determine the appropriate course of action. Please consult with a healthcare professional for proper assessment and guidance."
        
        for key, fallback_text in fallback_explanations.items():
            if key in diagnosis_lower:
                explanation = fallback_text
                break
        
        return {
            "explanation": explanation,
            "encyclopedia_context": "",
            "key_terms": [],
            "generation_method": "fallback",
            "success": False
        }
        """Create a more intelligent duration question based on AI-extracted context"""
        
        # If duration was already mentioned, acknowledge it
        if duration_hint:
            return f"You mentioned {symptom_type} for {duration_hint}. To better understand the timeline, which option best describes the duration?"
        
        # Customize question based on symptom type and characteristics
        if symptom_type in ['headache', 'head pain']:
            if any(char in ['pounding', 'throbbing'] for char in characteristics):
                return f"You're experiencing {symptom_type} with {', '.join(characteristics)} characteristics. How long have you had these symptoms?"
        
        # Default intelligent question
        if characteristics:
            return f"You described {symptom_type} with {', '.join(characteristics[:2])} qualities. How long have you been experiencing this?"
        else:
            return f"How long have you been experiencing this {symptom_type}?"
    def _extract_primary_symptom(self, collected_data: Dict[str, Any]) -> str:
        """Extract the primary symptom from collected data for context"""
        # Try AI-extracted data first
        ai_data = collected_data.get('ai_extracted_data', {})
        if ai_data.get('primary_symptom'):
            return ai_data['primary_symptom']
        
        # Fallback to rule-based extraction
        symptoms = collected_data.get('symptoms', '')
        if symptoms:
            return self._extract_symptom_type(symptoms)
        
        return 'symptoms'

    def _create_duration_question_with_ai_context(self, symptom_type: str, 
                                                 characteristics: List[str], 
                                                 duration_hint: str) -> str:
        """Create a more intelligent duration question based on AI-extracted context"""
        
        # If duration was already mentioned, acknowledge it
        if duration_hint:
            return f"You mentioned {symptom_type} for {duration_hint}. To better understand the timeline, which option best describes the duration?"
        
        # Create contextual question based on symptom type
        if symptom_type in ['headache', 'head pain']:
            if any(char in ['pounding', 'throbbing'] for char in characteristics):
                return f"You're experiencing {symptom_type} with {', '.join(characteristics)} characteristics. How long have you had these symptoms?"
        
        # Default intelligent question
        if characteristics:
            return f"You described {symptom_type} with {', '.join(characteristics[:2])} qualities. How long have you been experiencing this?"
        else:
            return f"How long have you been experiencing this {symptom_type}?"

    def _extract_symptom_type(self, symptoms: str) -> str:
        symptoms_lower = symptoms.lower()
        
        # Use regular expressions for better pattern matching
        pain_patterns = r'\b(pain|ache|hurt|sore|tender)\b'
        if re.search(pain_patterns, symptoms_lower):
            if any(word in symptoms_lower for word in ['knee', 'leg', 'joint']):
                return 'knee pain'
            elif any(word in symptoms_lower for word in ['head', 'headache']):
                return 'headache'
            elif any(word in symptoms_lower for word in ['back', 'spine']):
                return 'back pain'
            elif any(word in symptoms_lower for word in ['chest']):
                return 'chest pain'
            else:
                return 'pain'
        elif any(word in symptoms_lower for word in ['swelling', 'swollen', 'inflammation']):
            return 'swelling'
        elif any(word in symptoms_lower for word in ['rash', 'skin', 'itch', 'red']):
            return 'skin condition'
        elif any(word in symptoms_lower for word in ['nausea', 'vomit', 'stomach']):
            return 'digestive symptoms'
        elif any(word in symptoms_lower for word in ['cough', 'breathing', 'shortness']):
            return 'respiratory symptoms'
        else:
            return 'symptoms'
    
    def _assess_urgency(self, collected_data: Dict) -> str:
        """Assess urgency level based on collected data"""
        symptoms = collected_data.get('symptoms', '').lower()
        intensity = collected_data.get('intensity', '').lower()
        
        # High urgency indicators
        high_urgency_keywords = [
            'severe', 'unbearable', 'worst', 'emergency', 'can\'t', 'unable',
            'chest pain', 'breathing', 'blood', 'sudden', 'acute'
        ]
        
        if any(keyword in symptoms or keyword in intensity for keyword in high_urgency_keywords):
            return 'high'
        elif 'moderate' in intensity or 'significant' in symptoms:
            return 'moderate'
        else:
            return 'low'
    
    def process_image_analysis(self, session_id: str, image_analysis: str) -> Dict[str, Any]:
        """Process image analysis result with AI-powered symptom extraction"""
        session = self.get_session(session_id)
        session['collected_data']['image_analysis'] = image_analysis
        session['collected_data']['symptoms'] = image_analysis
        session['state'] = ConversationState.DURATION_INQUIRY
        
        # Use AI extraction for image analysis if available
        if self.use_ai_extraction:
            try:
                structured_data = asyncio.run(self._extract_symptoms_with_ai(image_analysis))
                session['collected_data']['ai_extracted_data'] = structured_data
                symptom_type = structured_data.get('primary_symptom', 'condition')
                
                logger.info(f"AI extracted from image analysis: {structured_data}")
            except Exception as e:
                logger.error(f"AI extraction failed for image analysis: {e}")
                symptom_type = self._extract_symptom_type(image_analysis)
        else:
            # Fallback to rule-based extraction
            symptom_type = self._extract_symptom_type(image_analysis)
        
        # Add to conversation history
        self.add_to_history(session_id, "Image uploaded", f"I can see {symptom_type} in your image.")
        
        return {
            "response_type": "multiple_choice", 
            "response_text": f"I can see what appears to be {symptom_type} in your image. To better understand your condition, how long have you been experiencing this?",
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
        # Add to conversation history
        session = self.get_session(session_id)
        self.add_to_history(session_id, transcription, "I heard your voice message.")
        
        return self.process_message(session_id, transcription)
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a comprehensive summary of the session"""
        session = self.get_session(session_id)
        
        # Calculate session duration
        created_at = session['created_at']
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        
        duration = datetime.now() - created_at
        
        return {
            'session_id': session_id,
            'current_state': session['state'].value if isinstance(session['state'], ConversationState) else str(session['state']),
            'collected_data': session['collected_data'],
            'conversation_length': len(session['conversation_history']),
            'session_duration_minutes': int(duration.total_seconds() / 60),
            'created_at': created_at.isoformat(),
            'last_activity': session['last_activity'].isoformat() if isinstance(session['last_activity'], datetime) else session['last_activity'],
            'total_sessions': len(self.sessions)
        }
    
    def reset_session(self, session_id: str) -> bool:
        """Reset a session to initial state"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session['state'] = ConversationState.INITIAL
            session['collected_data'] = {}
            session['last_activity'] = datetime.now()
            return True
        return False
    
    def get_all_sessions_summary(self) -> Dict[str, Any]:
        """Get summary of all active sessions"""
        self.cleanup_expired_sessions()
        
        return {
            'total_active_sessions': len(self.sessions),
            'sessions': [self.get_session_summary(sid) for sid in list(self.sessions.keys())[:10]]  # Limit to 10 for performance
        }