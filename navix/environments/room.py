# Copyright 2023 The Navix Authors.

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


from __future__ import annotations
from typing import Callable

import jax
import jax.numpy as jnp
from jax.random import KeyArray
from jax import Array

from navix import observations, tasks, terminations, graphics

from ..components import Goal, Player
from ..grid import random_positions, random_directions, room
from ..graphics import RenderingCache
from .environment import Environment, Timestep, State


class Room(Environment):
    @classmethod
    def create(
        cls,
        height: int,
        width: int,
        max_steps: int,
        gamma: float = 1.0,
        observation_fn: Callable[[State, RenderingCache], Array] = observations.rgb,
        reward_fn: Callable[[State, Array, State], Array] = tasks.navigation,
        termination_fn: Callable[
            [State, Array, State], Array
        ] = terminations.on_navigation_completion,
    ) -> Room:
        grid = room(height=height, width=width)
        return cls(
            max_steps=max_steps,
            gamma=gamma,
            observation_fn=observation_fn,
            reward_fn=reward_fn,
            termination_fn=termination_fn,
            cache=graphics.RenderingCache.init(grid),
        )

    def reset(self, key: KeyArray) -> Timestep:
        key, k1, k2 = jax.random.split(key, 3)

        # map
        positions = random_positions(k1, self.cache.grid, n=2)
        direction = random_directions(k2, n=1)
        # player
        player = Player(position=positions[0], direction=direction)
        # goal
        goal = Goal(position=positions[1][None])

        # systems
        state = State(
            key=key,
            player=player,
            goals=goal,
        )

        return Timestep(
            t=jnp.asarray(0, dtype=jnp.int32),
            observation=self.observation(state),
            action=jnp.asarray(0, dtype=jnp.int32),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            step_type=jnp.asarray(0, dtype=jnp.int32),
            state=state,
        )
