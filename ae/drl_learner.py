import numpy as np
import random
from collections import deque
try:
    from tensorflow.keras import models, layers, optimizers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. DRL will use simplified model.")

class DRLearner:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        if TF_AVAILABLE:
            self.model = self._build_keras_model()
        else:
            self.model = self._build_simple_model()
    
    def _build_keras_model(self):
        model = models.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=0.001))
        return model
    
    def _build_simple_model(self):
        # Simple model without TensorFlow dependency
        return SimpleModel(self.state_size, self.action_size)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        if TF_AVAILABLE:
            act_values = self.model.predict(state, verbose=0)
        else:
            act_values = self.model.predict(state)
            
        return np.argmax(act_values[0])
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                if TF_AVAILABLE:
                    target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
                else:
                    target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            
            if TF_AVAILABLE:
                target_f = self.model.predict(state, verbose=0)
                target_f[0][action] = target
                self.model.fit(state, target_f, epochs=1, verbose=0)
            else:
                self.model.update(state, action, target)
                
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        if TF_AVAILABLE:
            self.model.load_weights(name)

    def save(self, name):
        if TF_AVAILABLE:
            self.model.save_weights(name)


# Simple model implementation for environments without TensorFlow
class SimpleModel:
    def __init__(self, state_size, action_size):
        self.weights = np.random.rand(state_size, action_size) * 0.1
        
    def predict(self, state):
        return np.dot(state, self.weights)
        
    def update(self, state, action, target):
        # Simple update rule
        prediction = self.predict(state)
        error = target - prediction[0][action]
        self.weights[:, action] += 0.1 * error * state[0]