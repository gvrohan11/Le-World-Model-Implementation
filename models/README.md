## Model architecture

LeWM is built upong 2 components: encoder and predictor

Encoder: maps a frame observation (o_t) into a compact low-dim latent representation (z_t)

Encoder: z_t = Enc(o_t)

Predictor: models environment dynamics in latent space by predicting the embedding of the next frame observation \hat{z_{t + 1}} given the latent embedding z_t and action a_t

Predictor: \hat{z_{t + 1}} = Pred(z_t, a_t)

Loss(world model) = Loss(prediction) + \lambda * SIGReg(Z)

Loss(prediction) = abs(z_{t + 1} - \hat{z_{t + 1}})


## Vision vs World model

Vision model: takes an image -> produces a representation of what's in it (embedding)

World model: predicts. Inputs: current state, (sometimes) past state, action. Outputs: predicts the future state

*World model has a vision model inside it: LeWM has to see the present - so it has the encoder (vision model) that turns the frame into an embedding. Then a predictor that says "given this embedding + action, here's the next embedding". 

World model = encoder + predictor

We use the world model task to train the vision model. 