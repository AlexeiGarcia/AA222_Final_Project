"""
Animations for various dynamical systems using `matplotlib`.

Code has been edited by Jeremiah Montemayor

Courtesy of Autonomous Systems Lab (ASL), Stanford
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import matplotlib.animation as animation

import numpy as np

def animate_2D_quad(t, state, full_system = 1, frameDelay = 30):
    """Animate the planar quadrotor system from given position data.

       Input Arguments:
       state(n,8):  assumes the first four columns are x, z, θ, and ϕ
                    (i.e., x-position, z-position, pitch angle,
                    pendulum angle) in that order.

       full_system (boolean): true - quad+payload; false - quad only

       frameDelay (optional input): animation delay per frame in milliseconds

       Run the following lines to use this function:
       fig, ani = animate_2D_quad(t, state)
       ani.save("planar_quad_1.mp4", writer="ffmpeg")

    """
    # Extract the states
    x = state[:,0]
    z = state[:,1]
    θ = -state[:,2]

    if full_system:
      ϕ = state[:,3]
    else:
      ϕ = np.zeros(x.shape)

    # Geometry
    rod_width = 1.
    rod_height = 0.1
    axle_height = 0.2
    axle_width = 0.05
    prop_width = 0.5*rod_width
    prop_height = 1.5*rod_height
    hub_width = 0.3*rod_width
    hub_height = 2.5*rod_height
    pole_length = 1.
    pole_thick = 0.1
    ball_radius = 0.2

    # Figure and axis
    fig, ax = plt.subplots(dpi=100)
    x_min, x_max = np.min(x), np.max(x)
    x_pad = (rod_width + prop_width )/2 + 0.1*(x_max - x_min) + pole_length
    z_min, z_max = np.min(z), np.max(z)
    z_pad = (rod_width + prop_width)/2 + 0.1*(z_max - z_min) + pole_length
    ax.set_xlim([x_min - x_pad, x_max + x_pad])
    ax.set_ylim([z_min - z_pad, z_max + z_pad])
    ax.set_aspect(1.)

    # Quadrotor Artists
    rod = mpatches.Rectangle((-rod_width/2, -rod_height/2),
                             rod_width, rod_height,
                             facecolor='tab:red', edgecolor='k')
    hub = mpatches.FancyBboxPatch((-hub_width/2, -hub_height/2),
                                  hub_width, hub_height,
                                  facecolor='tab:red', edgecolor='k',
                                  boxstyle='Round,pad=0.,rounding_size=0.05')
    axle_left = mpatches.Rectangle((-rod_width/2, rod_height/2),
                                   axle_width, axle_height,
                                   facecolor='tab:red', edgecolor='k')
    axle_right = mpatches.Rectangle((rod_width/2 - axle_width, rod_height/2),
                                    axle_width, axle_height,
                                    facecolor='tab:red', edgecolor='k')
    prop_left = mpatches.Ellipse(((axle_width - rod_width)/2,
                                  rod_height/2 + axle_height),
                                 prop_width, prop_height,
                                 facecolor='tab:gray', edgecolor='k',
                                 alpha=0.7)
    prop_right = mpatches.Ellipse(((rod_width - axle_width)/2,
                                   rod_height/2 + axle_height),
                                  prop_width, prop_height,
                                  facecolor='tab:gray', edgecolor='k',
                                  alpha=0.7)

    patches = (rod, hub, axle_left, axle_right, prop_left, prop_right)

    # Pendulum Artists
    pole = mpatches.Arrow(0., 0., 0., -pole_length, width=pole_thick,
                              facecolor='tab:brown', edgecolor='k')
    ball = mpatches.Circle((0., -pole_length), radius=ball_radius,
                           facecolor='tab:orange', edgecolor='k')

    # Add patches to axis
    for patch in patches:
        ax.add_patch(patch)

    if full_system:
      ax.add_patch(pole)
      ax.add_patch(ball)

    # Trace and timestamp
    trace = ax.plot([], [], '--', linewidth=2, color='tab:blue')[0]
    timestamp = ax.text(0.1, 0.9, '', transform=ax.transAxes)

    # Animation
    def animate(k, t, x, z, θ, ϕ):

        # Quad rotation and translation
        transformQuad = mtransforms.Affine2D().rotate_around(0., 0., θ[k])
        transformQuad += mtransforms.Affine2D().translate(x[k], z[k])
        transformQuad += ax.transData
        for patch in patches:
            patch.set_transform(transformQuad)

        if full_system:
            # Pole translation and rotation
            transformPole = mtransforms.Affine2D().rotate_around(0., 0., ϕ[k])
            transformPole += mtransforms.Affine2D().translate(x[k], z[k])
            transformPole += ax.transData

            pole.set_transform(transformPole)

            # Payload translation and rotation
            transformBall = mtransforms.Affine2D().rotate_around(0., 0., ϕ[k])
            transformBall += mtransforms.Affine2D().translate(x[k], z[k])
            transformBall += ax.transData

            ball.set_transform(transformBall)

        # Trace and time stamp animation
        trace.set_data(x[:k+1], z[:k+1])
        timestamp.set_text('t = {:.1f} s'.format(t[k]))

        # Consolidate artists
        artists = patches + (trace, timestamp)
        if full_system:
            artists += (pole, ball)

        return artists

    ani = animation.FuncAnimation(fig, animate, t.size, fargs=(t, x, z, θ, ϕ),
                                  interval=frameDelay, blit=True)
    return fig, ani
