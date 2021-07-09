config = dict(
    agent=dict(),
    algo=dict(),
    env=dict(
        game="pong",
        num_img_obs=1,
    ),
    model=dict(),
    optim=dict(),
    runner=dict(
        n_steps=5e5,
        # log_interval_steps=1e5,
    ),
    sampler=dict(
        batch_T=10,
        batch_B=16,
        max_decorrelation_steps=1000,
    ),
)

configs = dict(
    default=config
)
