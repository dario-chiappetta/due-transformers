"""
Here we define a :class:`DialoGPTAgent` class that uses DialoGPT from Microsoft,
as it is implemented in the Transformers library. This is an instance of GPT2
that was pre-trained on a corpus of 147M conversation-like Reddit exchanges.

Here some resources:

* https://github.com/microsoft/DialoGPT
* https://github.com/huggingface/transformers/blob/master/model_cards/microsoft/DialoGPT-medium/README.md
* https://huggingface.co/transformers/model_doc/dialogpt.html

"""


from collections import defaultdict
from datetime import datetime

import torch
from transformers import AutoModelWithLMHead, AutoTokenizer
from due.agent import Agent
from due.event import Event

class DialoGPTAgent(Agent):
    
    def __init__(self, agent_id=None, model='microsoft/DialoGPT-medium', context_length=2):
        """
        Use Microsoft's DialoGPT, as it is implemented in the Transformers library.
        
        :param context_length: Number of previous conversation turns to consider for answer prediction
        :type context_length: `int`
        """
        super().__init__(agent_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelWithLMHead.from_pretrained(model)
        self.episode_history = defaultdict(list)
        self.context_length = context_length
        
    def save(self):
        raise NotImplementedError()
        
    def learn_episodes(self, episodes):
        raise NotImplementedError()
        
    def new_episode_callback(self, new_episode):
        pass
    
    def utterance_callback(self, episode):
        last_utterance = episode.last_event(Event.Type.Utterance).payload
        tokenized_utterance = self.tokenizer.encode(last_utterance + self.tokenizer.eos_token, return_tensors='pt')
        self.episode_history[episode.id].append(tokenized_utterance)
        model_input_ids = torch.cat(self.episode_history[episode.id][-1*self.context_length:], dim=-1)
        predicted_tokens = self.model.generate(model_input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)
        predicted_answer = self.tokenizer.decode(predicted_tokens[:, model_input_ids.shape[-1]:][0], skip_special_tokens=True)
        if predicted_answer:
            return [Event(Event.Type.Utterance, datetime.now(), self.id, predicted_answer)]
        return []
    
    def action_callback(self, episode):
        pass
    
    def leave_callback(self, episode):
        pass
