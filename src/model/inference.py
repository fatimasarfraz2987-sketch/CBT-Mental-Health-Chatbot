"""
Inference module for loading trained model and generating responses.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os


class DialogueInference:
    """Load model and generate therapy responses."""
    
    def __init__(self, model_path="models/fine_tuned/checkpoint-final"):
        """
        Load model and tokenizer.
        
        Args:
            model_path: Path to fine-tuned model checkpoint
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        self.model.eval()
    
    def generate_response(self, user_input, max_length=100, temperature=0.7):
        """
        Generate therapy response for user input.
        
        Args:
            user_input: User's question or message
            max_length: Maximum response length
            temperature: Sampling temperature (0-1)
            
        Returns:
            Generated therapy response
        """
        input_ids = self.tokenizer.encode(user_input, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1
            )
        
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response
    
    def batch_generate(self, user_inputs, max_length=100):
        """
        Generate responses for multiple inputs.
        
        Args:
            user_inputs: List of user messages
            max_length: Maximum response length
            
        Returns:
            List of generated responses
        """
        responses = []
        for user_input in user_inputs:
            response = self.generate_response(user_input, max_length)
            responses.append(response)
        return responses


if __name__ == "__main__":
    # Example usage
    inference = DialogueInference()
    
    user_query = "I'm feeling anxious about my job"
    response = inference.generate_response(user_query)
    print(f"User: {user_query}")
    print(f"Therapist: {response}")
