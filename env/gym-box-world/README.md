[How to create new environments for Gym](https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa)
# The BoxWorld environment
## Description
Box-World is a perceptually simple but combinatorially complex environment that requires abstractrelational reasoning and planning.

It consists of a 12×12 pixel room with keys and boxes randomly scattered. The room also contains an agent, represented by a single dark gray pixel, which can move in four directions:up,down,left,right.

Keys are represented by a single colored pixel. The agent can pick up a loose key (i.e., one notadjacent to any other colored pixel) by walking over it. Boxes are represented by two adjacent coloredpixels – the pixel on the right represents the box’s lock and its color indicates which key can be usedto open that lock; the pixel on the left indicates the content of the box which is inaccessible whilethe box is locked.

To collect the content of a box the agent must first collect the key that opens the box (the onethat matches the lock’s color) and walk over the lock, which makes the lock disappear. At this pointthe content of the box becomes accessible and can be picked up by the agent. Most boxes containkeys that, if made accessible, can be used to open other boxes. One of the boxes contains a gem,represented by a single white pixel. The goal of the agent is to collect the gem by unlocking thebox that contains it and picking it up by walking over it. Keys that an agent has in possession aredepicted in the input observation as a pixel in the top-left corner.

To see more in [Relational Deep Reinforcement Learning](https://arxiv.org/abs/1806.01830)
## Action mapping
- ``` UP``` - 0
- ``` DOWN``` - 1
- ``` LEFT``` - 2
- ``` RIGHT``` - 3

## Observations
- Screen:
    - ```frame```- 14x14 vectors of RGB pixels representing renderd screen.( The screen includes game boundaries)



## Scenarios
- BoxWoldEnv
    - Fixed the location of keys and boxes
- BoxWoldRandEnv
    - Randomly generate the location of keys and boxes
