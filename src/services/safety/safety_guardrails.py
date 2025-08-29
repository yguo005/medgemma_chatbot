import re
import logging
from typing import Dict, Any, List, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

class SafetyLevel(Enum):
    SAFE = "safe"
    EMERGENCY = "emergency"
    BLOCKED = "blocked"

class MedicalSafetyGuardrails:
    """
    Comprehensive safety guardrails for medical AI responses
    Based on the project requirements for responsible AI
    """
    
    def __init__(self):
        # Emergency keywords that require immediate medical attention
        self.emergency_keywords = {
            'chest pain', 'can\'t breathe', 'difficulty breathing', 'suicidal thoughts',
            'suicide', 'overdose', 'severe bleeding', 'heart attack', 'stroke',
            'severe allergic reaction', 'anaphylaxis', 'unconscious', 'seizure',
            'severe burns', 'choking', 'severe head injury'
        }
        
        # Forbidden diagnostic phrases that should never appear in responses
        self.forbidden_phrases = [
            'you have', 'the diagnosis is', 'this is definitely', 'you definitely have',
            'i diagnose you with', 'you are suffering from', 'the condition is',
            'you need to take', 'the treatment is', 'you should take medication'
        ]
        
        # Cautious language patterns we want to encourage
        self.cautious_patterns = [
            'could be related to', 'might indicate', 'sometimes suggests',
            'may be helpful to discuss', 'consider consulting', 'it would be wise to'
        ]
        
        # System prompt for responsible medical AI
        self.medical_system_prompt = """You are an AI Health Assistant. Your role is strictly informational and educational. You are NOT a doctor and you must NEVER provide a medical diagnosis.

Your primary goals are to help users articulate their symptoms and guide them to the appropriate next steps.

Follow these rules at all times:

1. **NEVER Diagnose:** Do not use definitive or diagnostic language. Never say "you have," "this is," or "the diagnosis is."

2. **USE CAUTIOUS LANGUAGE:** Always use probabilistic and cautious phrasing.
   - **Good:** "Symptoms like that *could be related to*...", "This *sometimes indicates*...", "*It may be helpful to discuss* these symptoms with a doctor."
   - **Bad:** "This is a sign of...", "You have a condition called..."

3. **MAINTAIN A CALM & SUPPORTIVE TONE:** Your tone should be neutral, reassuring, and professional. Avoid sensational or alarming language.

4. **FRAME AS POSSIBILITIES:** Frame all information as possibilities for the user to discuss with a real healthcare professional. The goal is to empower the user for their appointment, not to replace it.

5. **ALWAYS DEFER TO PROFESSIONALS:** Conclude every significant interaction by reinforcing the need to consult a doctor.

6. **PRIORITIZE SAFETY:** If the user's input contains keywords related to a medical emergency, immediately stop the standard flow and provide an emergency services directive.

Remember: You provide information to help users prepare for medical consultations, not to replace professional medical advice."""

    def check_emergency_keywords(self, user_input: str) -> Tuple[SafetyLevel, str]:
        """
        Check if user input contains emergency keywords requiring immediate action
        
        Args:
            user_input: User's message/symptoms
            
        Returns:
            Tuple of (SafetyLevel, emergency_message)
        """
        user_input_lower = user_input.lower()
        
        for keyword in self.emergency_keywords:
            if keyword in user_input_lower:
                emergency_message = self._generate_emergency_response(keyword)
                logger.warning(f"🚨 Emergency keyword detected: {keyword}")
                return SafetyLevel.EMERGENCY, emergency_message
        
        return SafetyLevel.SAFE, ""

    def _generate_emergency_response(self, keyword: str) -> str:
        """Generate appropriate emergency response based on keyword"""
        return f"""🚨 **EMERGENCY MEDICAL SITUATION DETECTED** 🚨

I notice you mentioned "{keyword}" which could indicate a serious medical emergency.

**PLEASE TAKE IMMEDIATE ACTION:**
• In the US: Call 911 immediately
• In other countries: Call your local emergency services
• Go to the nearest emergency room
• If possible, have someone with you

This AI assistant cannot help with medical emergencies. Please seek immediate professional medical attention.

Your safety is the top priority. Do not wait - get help now."""

    def validate_response(self, ai_response: str) -> Dict[str, Any]:
        """
        Validate AI response for safety violations
        
        Args:
            ai_response: Generated response from AI
            
        Returns:
            Dict with validation results and filtered response
        """
        issues = []
        filtered_response = ai_response
        
        # Check for forbidden diagnostic phrases
        response_lower = ai_response.lower()
        for phrase in self.forbidden_phrases:
            if phrase in response_lower:
                issues.append(f"Contains forbidden diagnostic phrase: '{phrase}'")
                # Replace with safer alternative
                filtered_response = self._replace_diagnostic_language(filtered_response, phrase)
        
        # Check for severity level (basic implementation)
        severity_score = self._calculate_severity_score(ai_response)
        
        # Add mandatory disclaimer if not present
        if not self._has_disclaimer(filtered_response):
            filtered_response = self._add_disclaimer(filtered_response)
        
        return {
            "is_safe": len(issues) == 0 and severity_score < 0.7,
            "issues": issues,
            "filtered_response": filtered_response,
            "severity_score": severity_score,
            "original_length": len(ai_response),
            "filtered_length": len(filtered_response)
        }

    def _replace_diagnostic_language(self, response: str, forbidden_phrase: str) -> str:
        """Replace forbidden diagnostic language with cautious alternatives"""
        replacements = {
            'you have': 'you might have symptoms that could be related to',
            'the diagnosis is': 'one possibility to discuss with a doctor is',
            'this is definitely': 'this could potentially be',
            'you definitely have': 'you may want to discuss symptoms of',
            'i diagnose you with': 'these symptoms might be related to',
            'you are suffering from': 'you may be experiencing symptoms that could indicate',
            'the condition is': 'one condition to consider discussing with a doctor is',
            'you need to take': 'a doctor might recommend',
            'the treatment is': 'potential treatments that doctors sometimes use include'
        }
        
        if forbidden_phrase in replacements:
            return response.replace(forbidden_phrase, replacements[forbidden_phrase])
        
        return response

    def _calculate_severity_score(self, response: str) -> float:
        """Calculate severity score based on alarming language"""
        severe_words = [
            'severe', 'critical', 'dangerous', 'life-threatening', 'emergency',
            'urgent', 'immediately', 'serious condition', 'fatal', 'death'
        ]
        
        response_lower = response.lower()
        severe_count = sum(1 for word in severe_words if word in response_lower)
        
        # Normalize by response length
        severity_score = severe_count / max(len(response.split()), 1)
        return min(severity_score, 1.0)

    def _has_disclaimer(self, response: str) -> bool:
        """Check if response contains appropriate medical disclaimer"""
        disclaimer_indicators = [
            'consult', 'doctor', 'healthcare professional', 'medical advice',
            'not a substitute', 'seek medical attention'
        ]
        
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in disclaimer_indicators)

    def _add_disclaimer(self, response: str) -> str:
        """Add medical disclaimer to response"""
        disclaimer = "\n\n*Please remember: This information is for educational purposes only and is not a substitute for professional medical advice. Always consult with a healthcare provider for proper diagnosis and treatment.*"
        return response + disclaimer

    def get_medical_system_prompt(self) -> str:
        """Get the complete system prompt for medical AI"""
        return self.medical_system_prompt

    def process_user_input(self, user_input: str) -> Dict[str, Any]:
        """
        Complete safety processing of user input
        
        Args:
            user_input: User's message/symptoms
            
        Returns:
            Dict with safety assessment and processed input
        """
        # Check for emergencies first
        safety_level, emergency_msg = self.check_emergency_keywords(user_input)
        
        if safety_level == SafetyLevel.EMERGENCY:
            return {
                "safety_level": safety_level,
                "should_block": True,
                "emergency_response": emergency_msg,
                "processed_input": user_input
            }
        
        return {
            "safety_level": SafetyLevel.SAFE,
            "should_block": False,
            "emergency_response": "",
            "processed_input": user_input
        }

    def create_safe_medical_prompt(self, user_query: str, retrieved_context: str) -> str:
        """
        Create a safe medical prompt combining system instructions, context, and user query
        
        Args:
            user_query: User's medical question
            retrieved_context: Retrieved medical knowledge from RAG
            
        Returns:
            Complete prompt with safety guardrails
        """
        safe_prompt = f"""{self.medical_system_prompt}

**Medical Knowledge Context:**
{retrieved_context}

**User's Question:**
{user_query}

**Instructions:**
Based on the medical knowledge provided above, give a helpful, informative response about the user's symptoms or question. Remember to:
1. Use cautious, probabilistic language
2. Suggest consulting healthcare professionals
3. Avoid definitive diagnoses
4. Be supportive and informative
5. Include appropriate disclaimers

**Response:**"""
        
        return safe_prompt