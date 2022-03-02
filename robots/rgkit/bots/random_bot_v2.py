#!/usr/bin/python
# coding=utf-8

"""
brief:          Random Bot - Either Attacks weakest Enemy, Guards, Moves, or if needed, Suicides near Enemy
author:         dove-zp
contact:        https://github.com/dove-zp/robot-game
version:        2021/SEP/04
license:        MIT License
"""

import random
from rgkit import rg


# --------------------------------------------------------------------------------------------------


class Robot(object):

    def __init__(self):
        self.rnd = random.Random()

    @staticmethod
    def is_location_occupied(game, location=(0, 0)):
        if location in game.robots.keys():
            return True
        return False

    def is_ally_at(self, game, location=(0, 0)):
        if self.is_location_occupied(game, location):
            return game.robots[location].player_id == self.player_id
        return False

    def is_enemy_at(self, game, location=(0, 0)):
        if self.is_location_occupied(game, location):
            return game.robots[location].player_id != self.player_id
        return False

    def most_vulnerable_adjacent_enemy(self, game):
        enemies = []
        if self.is_enemy_at(game, self.right()):
            enemies.append(self.right())
        if self.is_enemy_at(game, self.left()):
            enemies.append(self.left())
        if self.is_enemy_at(game, self.up()):
            enemies.append(self.up())
        if self.is_enemy_at(game, self.down()):
            enemies.append(self.down())
        enemies.sort(key=lambda x: game.robots[x].hp)
        if enemies:
            return enemies[0]
        return None

    def right(self):
        return (self.location[0] + 1, self.location[1])

    def left(self):
        return (self.location[0] - 1, self.location[1])

    def up(self):
        return (self.location[0], self.location[1] - 1)

    def down(self):
        return (self.location[0], self.location[1] + 1)

    def guard(self, game):
        return self.suicide(game)

    def suicide(self, game):
        enemy = self.most_vulnerable_adjacent_enemy(game)
        if self.hp <= rg.settings.suicide_damage and enemy is not None:
            return ['suicide']
        return ['guard']

    def attack(self, game):
        enemy = self.most_vulnerable_adjacent_enemy(game)
        if enemy is not None:
            return ['attack', enemy]
        return self.move(game)

    def move(self, game):
        locations = rg.locs_around(self.location, filter_out=('invalid', 'obstacle', 'spawn'))
        if len(locations) > 0:
            open_spaces = []
            for l in locations:
                if not self.is_ally_at(game, l):
                    open_spaces.append(l)
            if len(open_spaces) > 0:
                return ['move', rg.toward(self.location, self.rnd.choice(open_spaces))]
        return self.suicide(game)

    def move_center(self):
        return ['move', rg.toward(self.location, rg.CENTER_POINT)]

    def act(self, game):
        self.rnd.seed(game.seed)
        if 'spawn' in rg.loc_types(self.location):
            return self.move_center()
        return self.rnd.choice([self.guard(game), self.move(game), self.attack(game)])
