import numpy as np
import gym, time
from collections import deque
from keras.models import load_model
from keras import backend as K
from skimage import color


def huber_loss(a, b):
    e = a - b
    quad = e**2 / 2
    lin = abs(e) - 1/2
    use_lin = (abs(e) > 1.0)
    use_lin = K.cast(use_lin, 'float32')
    return use_lin * lin + ( 1 - use_lin) * quad

def to_gray(img):
    # Convert images to grayscale with values between 0 and 1
    return color.rgb2gray(img).astype(np.float32)

def downsample(img):
    # Downsampling an image for faster computing
    return img[::2, ::2]

def preprocessing(obs):
    radar = obs[3:36,74:96]
    scene = obs[38:178,8:]
    radar = to_gray(radar)
    scene = downsample(to_gray(scene))
    return radar, scene


def choose_action(model, stacked_radars, stacked_scenes):
    # Choose the best action according to the model's prediction
    
    #Initializing
    radars = list(stacked_radars)
    scenes = list(stacked_scenes)
        
    # Reshaping for the model to use (typically 4 frames of each type, as the model has 4 input channels)
    radars = np.reshape(radars, (batch_size, radars[0].shape[0], radars[0].shape[1], n_frames))
    scenes = np.reshape(scenes, (batch_size, scenes[0].shape[0], scenes[0].shape[1], n_frames))
    
    prediction = model.predict([scenes, radars, np.ones((1,n_actions))])
    
    return np.argmax(prediction)


def stack_frames(stacked_radars, stacked_scenes, observation):
    radar, scene = preprocessing(observation)
    if (len(stacked_radars) < n_frames):
        for _ in range(n_frames):
            stacked_radars.append(radar)
            stacked_scenes.append(scene)
    else:
        stacked_radars.append(radar)
        stacked_scenes.append(scene)
    return stacked_radars, stacked_scenes

def run_game(n_games, model):
    
    rew_max = 0
    
    for _ in range(n_games):
        
        rew_total = 0
        env.reset()
        action = 0
        
        done = False
        is_terminal = [False] * n_actions
        
        stacked_scenes = deque([], maxlen=n_frames)
        stacked_radars = deque([], maxlen=n_frames)
        
        while not done :
            env.render()
            observation, rew, done, info = env.step(action)
            
            stacked_radars, stacked_scenes = stack_frames(stacked_radars,
                                                          stacked_scenes,
                                                          observation)
            
            action = choose_action(model, stacked_radars, stacked_scenes)
            time.sleep(0.02)

            rew_total += rew
            
        rew_max = max(rew_max, rew_total)
        env.close()
    
    return rew_max

if __name__ == "__main__":

	model = load_model("trained_model", custom_objects={'huber_loss': huber_loss})

	env = gym.make("BattleZone-v0")

	n_games = 3
	n_actions = 18
	n_frames = 2
	batch_size = 1

	run_game(n_games, model)
