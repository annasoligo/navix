from dataclasses import dataclass, field
import tyro
import jax
import numpy as np
import jax.numpy as jnp
import navix as nx
from navix import observations
from navix.agents import PPO, PPOHparams, ActorCritic
from navix.environments.environment import Environment
from navix.transitions import deterministic_transition

import matplotlib.backends.backend_agg as agg
import pygame
from pygame.locals import *
import numpy as np


COLORS = {
        'background': (255, 255, 255),  # White background
        'grid_background': (0, 0, 0),   # Black grid interior
        'grid': (50, 50, 50),           # Dark grey grid
        'border': (100, 100, 100),      # Light grey border
        'player': (255, 0, 0),          # Red player
        'goal': (0, 255, 0),            # Green goal
        'ball': (0, 0, 255),            # Blue ball
        'key': (255, 255, 0),           # Yellow key
        'door': (165, 42, 42),          # Brown door
        'wall': (100, 100, 100),        # Grey wall
        'text': (0,0,0),                # Black Text
        'highlight': (0, 100, 0)        # Dark green highlight
    }

border_width = 80
grid_bottom = 400 + 2*border_width
line_height = 30

def get_env_actions(env):
    actions = env.action_set
    print(actions)
    if isinstance(actions, tuple):
        if isinstance(actions[0], tuple):
            actions = actions[0]
        actions = [func.__name__ for func in actions]
    return actions

def draw_key(screen, x, y, cell_size, color):
    key_width = cell_size // 5
    key_height = cell_size //2
    handle_radius = cell_size // 5
    
    # Draw the key shaft
    pygame.draw.rect(screen, color, 
                     (x - key_width // 2, y - (key_height // 2 - handle_radius // 2), 
                      key_width, key_height))
    
    # Draw the key handle (circle)
    pygame.draw.circle(screen, color, 
                       (x, y - (key_height // 2 - handle_radius // 2)), 
                       handle_radius)
    
    # Draw key teeth (small rectangles) 
    teeth_width = key_width // 2
    teeth_height = key_height // 6
    for i in range(2):
        pygame.draw.rect(screen, COLORS['grid_background'],
                         (x + key_width // 2 - int(teeth_width*0.9),
                          y + key_height // 4 + handle_radius // 2 + (i-1) * int(teeth_height * 1.5),
                          teeth_width, teeth_height))

def draw_grid(screen, entities, grid_size=6):
    
    # Clear the screen
    screen.fill(COLORS['background'])

    # Draw border
    pygame.draw.rect(screen, COLORS['border'], (0, 0, 400 + 2*border_width, 400 + 2*border_width), border_width)

    # Draw grid interior
    pygame.draw.rect(screen, COLORS['grid_background'], (border_width, border_width, 400, 400))
    
    # Draw grid lines
    cell_size = 400 // grid_size
    for i in range(grid_size + 1):
        pygame.draw.line(screen, COLORS['grid'], (border_width, i * cell_size + border_width), 
                         (400 + border_width, i * cell_size + border_width), width=2)
        pygame.draw.line(screen, COLORS['grid'], (i * cell_size + border_width, border_width), 
                         (i * cell_size + border_width, 400 + border_width), width=2)
    
    # Draw entities
    for entity_type, entity_data in entities.items():
        if entity_type == 'player':
            pos = entity_data.position[0] - 1
            # Draw player as a smaller diamond
            size = cell_size // 4  # Reduced size
            center_x = pos[1] * cell_size + cell_size // 2 + border_width
            center_y = pos[0] * cell_size + cell_size // 2 + border_width
            direction = entity_data.direction
            if direction == 3:
                triangle_points = [
                    (center_x, center_y - size),  # top
                    (center_x + size, center_y + size),  # right
                    (center_x - size, center_y + size)  # left
                ]
            elif direction == 0:
                triangle_points = [
                    (center_x + size, center_y),  # right
                    (center_x - size, center_y - size),  # top
                    (center_x - size, center_y + size)  # bottom
                ]
            elif direction == 1:
                triangle_points = [
                    (center_x, center_y + size),  # bottom
                    (center_x + size, center_y - size),  # right
                    (center_x - size, center_y - size)  # left
                ]
            elif direction == 2:
                triangle_points = [
                    (center_x - size, center_y),  # left
                    (center_x + size, center_y - size),  # top
                    (center_x + size, center_y + size)  # bottom
                ]
            pygame.draw.polygon(screen, COLORS['player'], triangle_points)  

        elif entity_type == 'ball' or entity_type == 'balls':
            positions = entity_data.position - 1
            for pos in positions:
                # Draw balls as circles
                center_x = pos[1] * cell_size + cell_size // 2 + border_width
                center_y = pos[0] * cell_size + cell_size // 2 + border_width
                radius = cell_size // 3
                pygame.draw.circle(screen, COLORS['ball'], (center_x, center_y), radius)
        elif entity_type == 'key':
            pos = entity_data.position[0] - 1
            center_x = pos[1] * cell_size + cell_size // 2 + border_width
            center_y = pos[0] * cell_size + cell_size // 2 + border_width
            draw_key(screen, center_x, center_y, cell_size, COLORS['key'])
        
        elif entity_type in COLORS:
            positions = entity_data.position - 1
            for pos in positions:
                pygame.draw.rect(screen, COLORS[entity_type], 
                                 (pos[1] * cell_size + border_width, pos[0] * cell_size + border_width, cell_size, cell_size))
    
    # Update display
    pygame.display.flip()

def run_interactive_animation(env, seed=0):
    
    pygame.init()
    screen = pygame.display.set_mode((800, 800))
    
    running = True
    paused = False
    seed = seed
    key = jax.random.PRNGKey(seed)
    obs = env.reset(key)
    step = jax.jit(env.step)

    draw_grid(screen, obs.state.entities, env.height-2)
    
    run_len = 0
    ret = 0
    
    font = pygame.font.Font(None, 24)
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT:
                    action = 1
                elif event.key == pygame.K_LEFT:
                    action = 0
                elif event.key == pygame.K_UP:
                    action = 2
                else:
                    pass

                
                next_obs = step(obs, action)
                new_done = next_obs.is_done() | (run_len >= env.max_steps - 1)
                
                if new_done:
                    key, reset_key = jax.random.split(key)
                    next_obs = env.reset(reset_key)
                    
                obs = next_obs
                draw_grid(screen, obs.state.entities, env.height-2)

                # Controls text at bottom
                controls_text = "Space = Pause, L = Left, U = Up, R = Right"
                controls_render = font.render(controls_text, True, COLORS['border'])
                screen.blit(controls_render, (10, grid_bottom + 10))
                
                pygame.display.flip()

    
    pygame.quit()

if __name__ == "__main__":

    env_id = "Navix-Dynamic-Obstacles-6x6-v0"

    env = nx.make(
        env_id,
        observation_fn=observations.symbolic_first_person,
        # transitions_fn=deterministic_transition,          # UNCOMMENT FOR STATIC BALLS
    )
    run_interactive_animation(env)
