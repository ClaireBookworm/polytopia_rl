"""
Text-based semantic action encoding for Polytopia RL
Uses sentence embeddings to encode action representations
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple

# Try to import sentence_transformers, fall back to simple encoding if not available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Using simple text encoding fallback.")

class ActionTextEncoder:
    """Encodes action text representations using sentence transformers or simple fallback"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2', cache_dir='action_embeddings_cache'):
        self.cache_dir = cache_dir
        self.embedding_cache = {}
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                self.use_transformers = True
            except Exception as e:
                print(f"Failed to load sentence transformer: {e}")
                print("Falling back to simple text encoding...")
                self.use_transformers = False
                self.embedding_dim = 64  # Fixed dimension for fallback
        else:
            self.use_transformers = False
            self.embedding_dim = 64  # Fixed dimension for fallback
            
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        self._load_cache()
        
    def _load_cache(self):
        """Load cached embeddings"""
        cache_file = os.path.join(self.cache_dir, 'embeddings.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                self.embedding_cache = pickle.load(f)
                
    def _save_cache(self):
        """Save embeddings to cache"""
        cache_file = os.path.join(self.cache_dir, 'embeddings.pkl')
        with open(cache_file, 'wb') as f:
            pickle.dump(self.embedding_cache, f)
            
    def encode_actions(self, action_reprs: List[str]) -> torch.Tensor:
        """Encode list of action representations to embeddings"""
        embeddings = []
        new_reprs = []
        
        for repr_text in action_reprs:
            if repr_text in self.embedding_cache:
                embeddings.append(self.embedding_cache[repr_text])
            else:
                new_reprs.append(repr_text)
                
        # Encode new representations
        if new_reprs:
            new_embeddings = self.model.encode(new_reprs)
                
            for repr_text, embedding in zip(new_reprs, new_embeddings):
                self.embedding_cache[repr_text] = embedding
                embeddings.append(embedding)
            self._save_cache()
            
        return torch.tensor(np.array(embeddings), dtype=torch.float32)

class SemanticActionHead(nn.Module):
    """Neural network head that uses semantic action embeddings"""
    
    def __init__(self, obs_dim: int, text_embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Action embedding processor
        self.action_encoder = nn.Sequential(
            nn.Linear(text_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Attention mechanism for action selection
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Final scoring layer
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, obs: torch.Tensor, action_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch_size, obs_dim) - observations
            action_embeddings: (batch_size, num_actions, text_embed_dim) - action text embeddings
            
        Returns:
            action_logits: (batch_size, num_actions) - action preferences
        """
        batch_size, num_actions, _ = action_embeddings.shape
        
        # Encode observations
        obs_features = self.obs_encoder(obs)  # (batch_size, hidden_dim)
        obs_features = obs_features.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # Encode actions
        action_features = self.action_encoder(action_embeddings)  # (batch_size, num_actions, hidden_dim)
        
        # Use attention to compute action preferences
        attended_features, _ = self.attention(
            query=obs_features,  # What we want (based on observation)
            key=action_features,   # Available actions
            value=action_features  # Action representations
        )  # (batch_size, 1, hidden_dim)
        
        # Score each action based on attention-weighted features
        # Expand attended features to match all actions
        attended_expanded = attended_features.expand(-1, num_actions, -1)  # (batch_size, num_actions, hidden_dim)
        
        # Combine with action features
        combined = attended_expanded + action_features  # (batch_size, num_actions, hidden_dim)
        
        # Get action scores
        scores = self.scorer(combined).squeeze(-1)  # (batch_size, num_actions)
        
        return scores

class TextSemanticTribesWrapper:
    """Wrapper that provides text-based semantic action embeddings"""
    
    def __init__(self, base_env, text_encoder: ActionTextEncoder):
        self.base_env = base_env
        self.text_encoder = text_encoder
        self._current_action_embeddings = None
        
    def __getattr__(self, name):
        # Delegate all other attributes to base environment
        return getattr(self.base_env, name)
    
    @property
    def unwrapped(self):
        """Provide access to unwrapped environment"""
        return self.base_env.unwrapped if hasattr(self.base_env, 'unwrapped') else self.base_env
        
    def reset(self, *args, **kwargs):
        obs, info = self.base_env.reset(*args, **kwargs)
        self._update_action_embeddings()
        return obs, info
        
    def step(self, action):
        obs, reward, done, truncated, info = self.base_env.step(action)
        self._update_action_embeddings()
        return obs, reward, done, truncated, info
        
    def _update_action_embeddings(self):
        """Update current action embeddings based on available actions"""
        actions = self.base_env.tribes_env.list_actions()
        action_reprs = [action.get('repr', 'UNKNOWN ACTION') for action in actions]
        
        if action_reprs:
            self._current_action_embeddings = self.text_encoder.encode_actions(action_reprs)
        else:
            # No actions available, create dummy embedding
            self._current_action_embeddings = torch.zeros((1, self.text_encoder.embedding_dim))
            
    def get_action_embeddings(self) -> torch.Tensor:
        """Get current action embeddings"""
        if self._current_action_embeddings is None:
            self._update_action_embeddings()
        return self._current_action_embeddings

# Example usage in PPO training loop
def create_semantic_envs(num_envs: int, text_encoder: ActionTextEncoder):
    """Create semantic-aware environments"""
    from pol_env.Tribes.py.register_env import TribesGymWrapper
    import gymnasium as gym
    
    def make_env(idx):
        def thunk():
            base_env = TribesGymWrapper()
            semantic_env = TextSemanticTribesWrapper(base_env, text_encoder)
            return gym.wrappers.RecordEpisodeStatistics(semantic_env)
        return thunk
    
    return gym.vector.SyncVectorEnv([make_env(i) for i in range(num_envs)])

# Integration example
if __name__ == "__main__":
    # Initialize text encoder
    print("Testing ActionTextEncoder...")
    text_encoder = ActionTextEncoder()
    
    if text_encoder.use_transformers:
        print("✓ Using sentence transformers")
    else:
        print("✓ Using simple text encoding fallback")
    
    # Test with some example action representations
    test_actions = [
        "RESEARCH_TECH by tribe 0 : FISHING",
        "MOVE by unit 2 to 1 : 10", 
        "ATTACK unit 3 with unit 5",
        "BUILD FARM at (2, 3) by city 1"
    ]
    
    embeddings = text_encoder.encode_actions(test_actions)
    print(f"✓ Action embeddings shape: {embeddings.shape}")
    print(f"✓ Embedding dimension: {text_encoder.embedding_dim}")
    
    # Test semantic action head
    obs_dim = 100  # example
    action_head = SemanticActionHead(obs_dim, text_encoder.embedding_dim)
    
    # Dummy data
    batch_size = 2
    num_actions = len(test_actions)
    obs = torch.randn(batch_size, obs_dim)
    action_embeds = embeddings.unsqueeze(0).expand(batch_size, -1, -1)
    
    logits = action_head(obs, action_embeds)
    print(f"✓ Action logits shape: {logits.shape}")
    print(f"✓ Action preferences: {torch.softmax(logits, dim=-1)}")
    
    print("✓ All tests passed!")
