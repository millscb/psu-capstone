import turtle


def rolling_ball():

    while True:

        turtle.speed(1)
        turtle.penup()
        turtle.goto(0, 0)
        turtle.pendown()


if __name__ == "__main__":
    rolling_ball()
