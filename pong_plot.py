# Import the pygame library and initialise the game engine
from typing import Type
import pygame
from datetime import datetime
from copy import deepcopy
from pygame.mixer import pause
from interfaces.model_interface import (
    Model,
    feedforward,
    freeModel,
    getMutated,
    train_from_array,
    save_model,
    load_model,
    createBitmap,
)
from objects import Paddle, Ball
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

pygame.init()

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Open a new window
size = (int(700), int(500))
# scale factors
sfx = float(size[0]) / 700.0
sfy = float(size[1]) / 500.0
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Pong")

padA = Paddle(sfy, WHITE, 10 * sfx, 100 * sfy)
padB = Paddle(sfy, WHITE, 10 * sfx, 100 * sfy)
ball = Ball(sfx, sfy, WHITE, 10 * sfx, 10 * sfy, [-5, 0])


def reset_paddles_ball():
    padA.rect.x = 20 * sfx
    padA.rect.y = 200 * sfy
    padB.rect.x = 670 * sfx
    padB.rect.y = 200 * sfy
    ball.rect.x = 345 * sfx
    ball.rect.y = 245 * sfy


reset_paddles_ball()

# This will be a list containing all the sprites we intend to use in my game.
all_sprites_list = pygame.sprite.Group()

# Add all the paddles and the balls to the list of sprites
all_sprites_list.add(padA)
all_sprites_list.add(padB)
all_sprites_list.add(ball)


def plotScores():
    if len(time_per_cycle) <= 1 and len(top_score_ratio) <= 1:
        return
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.set_title("Performance of an ANN trained with backpropagation and RL")
    ax1.set_xlabel(f"Cycles of {gamespacer} games")
    ax1.set_ylabel("Score ratio per cycle")
    ax1.grid(True)
    ax1.set_ylim(0, 1)
    ax1.set_xlim(1, len(top_score_ratio))
    # ax1.set_xlim(0, len(top_score_ratio) + 1)
    ax1.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    if len(top_score_ratio) < 10:
        ax1.locator_params(axis="x", nbins=len(top_score_ratio))
    # ax1.set_xticks([*range(1, len(top_score_ratio) + 1)])
    ax1.fill_between([*range(1, len(top_score_ratio) + 1)], top_score_ratio)

    ax2 = fig.add_subplot(212)
    ax2.set_title("Duration (total: %d sec)" % sum(time_per_cycle))
    ax2.set_xlabel(f"Cycles of {gamespacer} games")
    ax2.set_ylabel("Duration per cycle in seconds")
    ax2.grid(True)
    ax2.set_xlim(1, len(time_per_cycle))
    # ax2.set_xlim(0, len(time_per_cycle) + 1)
    ax2.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%d"))
    if len(time_per_cycle) < 10:
        ax2.locator_params(axis="x", nbins=len(time_per_cycle))
    # ax2.set_xticks([*range(1, len(time_per_cycle) + 1)])
    ax2.plot(
        [*range(1, len(time_per_cycle) + 1)],
        time_per_cycle,
        "b",
    )
    ax2.plot(
        [*range(1, len(time_per_cycle) + 1)],
        [
            sum(time_per_cycle) / len(time_per_cycle)
            for _ in range(len(time_per_cycle))
        ],
        "m--",
    )
    ax2.annotate(
        "%.2f" % (sum(time_per_cycle) / len(time_per_cycle)),
        (1, sum(time_per_cycle) / len(time_per_cycle)),
        textcoords="offset points",
        xytext=(0, 5),
        ha="left",
    )
    fig.tight_layout()

    fig.savefig("Data/RL/" + fname + ".png")
    plt.close(fig)
    plt.close()
    plt.clf()
    del fig


# The loop will carry on until the user exit the game
carryOn = True

# The clock will be used to control how fast the screen updates
clock = pygame.time.Clock()

# Initialise player scores
spd = 5

# Initialize models
seed = 2
# shape = [size[0] * size[1], 24, 1]
shape = [4, 2, 1]
actFuncts = [3, 2]
model1 = Model(seed, len(shape), shape, actFuncts)
# load_model(model1, "saves/model1-2021-10-08_07-41-36.dat")

model1_tracker = []
model1_training = []
model1_training_total = []

top_score_ratio = []
time_per_cycle = []
t0 = datetime.now()


def save(m1, top=False):
    filename_end = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if top:
        filename_end += "_TOP"
    save_model(m1, "saves/model1-" + filename_end + ".dat")
    print("Saved")


dScore = 0
counter = 0
trainings = 0
usrPlay = False
gamecounter = 0
spacer = 10
gamespacer = 100
saverequirementratio = 0.95
speedup = False
draw_screen = True
learning = True
hard = False
topOnly = False
its = 1
lr = 1
dataThinner = 1  # 2^x times smaller td
name = "_seed%d_s" % seed
for s in shape:
    name += str(s) + "-"
name = name[:-1] + "_a"
for a in actFuncts:
    name += str(a) + "-"
name = name[:-1] + "_lr%f" % lr

score1 = 0
score2 = 0

fname = t0.strftime("%Y-%m-%d_%H-%M-%S") + name
print(fname)

with open("Data/RL/" + fname + "_LOG.txt", "a+") as f:
    f.write(
        name
        + "\n"
        + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + "\t"
        + str(trainings)
        + f"\tdata thinner = {dataThinner}, games per cycle = {gamespacer}"
        + (", handicapped\n" if not hard else "\n")
    )
# -------- Main Program Loop -----------
while carryOn:
    counter += 1
    # --- Main event loop
    for event in pygame.event.get():  # User did something
        if event.type == pygame.QUIT:  # If user clicked close
            carryOn = False  # Flag that we are done so we exit this loop
        elif event.type == pygame.KEYDOWN:
            if (
                event.key == pygame.K_x
            ):  # Pressing the x Key will quit the game
                carryOn = False
                plotScores()
            elif event.key == pygame.K_SPACE:
                speedup = not speedup
                print("speedup: ", speedup)
            elif event.key == pygame.K_t:
                topOnly = not topOnly
                print("topOnly: ", topOnly)
            elif event.key == pygame.K_1:
                draw_screen = not draw_screen
                print("draw_screen: ", draw_screen)
            elif event.key == pygame.K_l:
                learning = not learning
                print("learning: ", learning)
            elif event.key == pygame.K_s:
                save(model1)
            elif event.key == pygame.K_2:
                hard = not hard
                print("hard: ", hard)
            elif event.key == pygame.K_u:
                usrPlay = not usrPlay
                print("usrPlay: ", usrPlay)
            elif event.key == pygame.K_p:
                print("plotted")
                plotScores()

    # --- Game logic should go here
    all_sprites_list.update()

    # Get input data for model
    input_data = [
        ball.rect.x / size[0],
        ball.rect.y / size[1],
        (padA.rect.y + 100 * sfy / 2) / size[1],
        # (padB.rect.y + 100 * sfy / 2) / size[1],
        # ball.velocity[0] / 5,
        ball.velocity[1],
    ]

    # objects[index][4] = feedforward(model1, input_data)
    out = feedforward(model1, input_data)[0]

    if learning:
        model1_tracker.append(
            [
                input_data,
                [float(int(out + 0.5))],
            ]
        )

    mv1 = 0
    mv2 = 0
    # Moving the paddles according to the models

    if out <= 0.5:  # objects[index][4][0] >= 0.5:
        mv1 = -spd
        padA.moveUp(spd * sfy)
    if out > 0.5:  # objects[index][4][0] < 0.5:
        mv1 = spd
        padA.moveDown(spd * sfy)
    if counter % (spacer) == 0 or hard:
        if ball.rect.y < padB.rect.y + 100 * sfy / 2:
            mv2 = -spd
            padB.moveUp(spd * sfy)
        if ball.rect.y > padB.rect.y + 100 * sfy / 2:
            mv2 = spd
            padB.moveDown(spd * sfy)

    # Detect collisions between the ball and the paddles
    if pygame.sprite.collide_mask(ball, padA):
        ball.bounce(mv1)
    if pygame.sprite.collide_mask(ball, padB):
        ball.bounce(mv2)

    # Check if the ball is bouncing against any of the 4 walls:
    if ball.rect.x >= (690 - 20) * sfx:
        reset_paddles_ball()
        for ele in model1_tracker:
            model1_training.append(ele)
        model1_tracker = []
        score1 += 1
        dScore += 1
        ball.velocity[0] = (gamecounter % 2) * -10 + 5
        ball.velocity[1] = 0
        gamecounter += 1
    if ball.rect.x <= 20 * sfx:
        reset_paddles_ball()
        for ele in model1_tracker:
            model1_training.append([ele[0], [float(not bool(ele[1][0]))]])
        model1_tracker = []
        score2 += 1
        ball.velocity[0] = (gamecounter % 2) * -10 + 5
        ball.velocity[1] = 0
        gamecounter += 1
    if ball.rect.y > 490 * sfy:
        ball.velocity[1] = -ball.velocity[1]
    if ball.rect.y <= 0:
        ball.velocity[1] = max(-ball.velocity[1], 1)

    if gamecounter == gamespacer:
        gamecounter = 0
        if learning:
            print(
                dScore,
                "A: %d\t B: %d\t Training..." % (score1, score2),
                end="\r",
            )
            trainings += 1
            top_score_ratio.append(dScore / gamespacer)

            t1 = datetime.now()
            diff = t1 - t0
            time_per_cycle.append(diff.total_seconds())
            t0 = t1

            with open("Data/RL/" + fname + "_LOG.txt", "a+") as f:
                f.write(
                    datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    + "\t"
                    + str(trainings)
                    + "\tduration: "
                    + str(diff.total_seconds())
                    + "\t["
                    + str(dScore)
                    + "]\n",
                )
            if dScore > int(gamespacer * saverequirementratio):
                save(model1, True)
                createBitmap(model1, fname + ".bmp")
                plotScores()
                exit(0)
            for _ in range(dataThinner - 1):
                for i in range(0, int(len(model1_training) / 2)):
                    model1_training.pop(i)
            train_from_array(
                model1, its, lr, model1_training, 1000000, 0, seed
            )
            createBitmap(model1, "images/" + fname + f"-{trainings}.bmp")
            model1_training = []
            print(dScore, "A: %d\t B: %d\t Done...\t\t\t" % (score1, score2))
        dScore = 0
    # if counter % save_spacer == 0:
    #    save(model1)

    # --- Drawing code should go here
    if draw_screen:
        # First, clear the screen to black.
        screen.fill(BLACK)
        # Draw the net
        pygame.draw.line(
            screen,
            WHITE,
            [349 * sfx, 0],
            [349 * sfx, 500 * sfy],
            int(10 * sfy),
        )

        # Now let's draw all the sprites in one go.
        all_sprites_list.draw(screen)

        # Display scores:
        font = pygame.font.Font(None, int(74 * sfx))
        text = font.render(str(score1), 1, WHITE)
        screen.blit(text, (250 * sfx, 10 * sfy))
        text = font.render(str(score2), 1, WHITE)
        screen.blit(text, (420 * sfx, 10 * sfy))

        # --- Go ahead and update the screen with what we've drawn.
        pygame.display.flip()

    # --- Limit to 60 frames per second
    # print("\bmv: " + str(mv) + "  ", end="\r")
    if not speedup:
        clock.tick(60)

# Once we have exited the main program loop we can stop the game engine:
pygame.quit()
