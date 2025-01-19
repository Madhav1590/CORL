import gym
import d4rl  # Ensure d4rl is imported to register its environments
import numpy as np
from pyvirtualdisplay import Display
from PIL import Image

# Start the virtual display
display = Display(visible=0, size=(1400, 900))
display.start()

# Load the environment
env = gym.make('halfcheetah-medium-v2')  # Replace with your environment

# Set the environment's state
state = np.array([0.0] * 17)  # Replace with your actual state of 17 values
env.reset()
env.env.sim.set_state_from_flattened(state)
env.env.sim.forward()

# Render the image
image = env.render(mode='rgb_array')

# Convert to an image and save/display
img = Image.fromarray(image)
img.save("rendered_state.png")
img.show()  # For local viewing; might not work on headless servers

# Close the environment
env.close()

# Stop the virtual display
display.stop()
