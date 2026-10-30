"""Compatibility factory using existing SERL DrQ and SmallEncoder implementations."""
def make_agent(seed,observations,action,image_keys,discount=.99):
    import jax
    import jax.numpy as jnp
    from einops import rearrange
    rearrange(jnp.zeros((1,1)), "a b -> b a")
    import flax.linen as nn
    from functools import partial
    from serl_launcher.agents.continuous.drq import DrQAgent
    from serl_launcher.common.encoding import EncodingWrapper
    from serl_launcher.vision.small_encoders import SmallEncoder
    from serl_launcher.networks.actor_critic_nets import Critic,Policy,ensemblize
    from serl_launcher.networks.mlp import MLP
    from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
    class RawImageEncoder(nn.Module):
        inner: nn.Module
        def __call__(self,observations,train=False,encode=True):
            if not encode:raise ValueError("Pre-encoded input is not supported by this raw RGB adapter")
            return self.inner(observations,train=train)
    encoders={k:RawImageEncoder(inner=SmallEncoder(features=(32,64,128,256),
        kernel_sizes=(3,3,3,3),strides=(2,2,2,2),padding="VALID",pool_method="avg",
        bottleneck_dim=256,spatial_block_size=8),name="encoder_"+k) for k in image_keys}
    encoder=EncodingWrapper(encoder=encoders,use_proprio=True,enable_stacking=True,image_keys=image_keys)
    network=dict(activations=nn.tanh,use_layer_norm=True,hidden_dims=[256,256],activate_final=True)
    backbone=ensemblize(partial(MLP,**network),10)(name="critic_ensemble")
    critic=Critic(encoder=encoder,network=backbone,name="critic")
    policy=Policy(encoder=encoder,network=MLP(**network),action_dim=7,
        tanh_squash_distribution=True,std_parameterization="exp",std_min=1e-5,std_max=5,name="actor")
    temperature=GeqLagrangeMultiplier(init_value=1e-2,constraint_shape=(),constraint_type="geq",name="temperature")
    return DrQAgent.create(jax.random.PRNGKey(seed),
        {k:jnp.asarray(x) for k,x in observations.items()},jnp.asarray(action),
        actor_def=policy,critic_def=critic,temperature_def=temperature,discount=discount,
        backup_entropy=False,critic_ensemble_size=10,critic_subsample_size=2,image_keys=image_keys)
