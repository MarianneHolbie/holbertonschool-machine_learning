#!/usr/bin/env python3
"""
    Module For prompt Q: and prints A: as a response
"""

quit_list = ["exit", "quit", "goodbye", "bye"]


def loop_QaBot():
    """
        Loop to capture question and print answer
        or quit if word in quit_list
    """
    while True:
        question = input("Q: ")
        if question.lower() in quit_list:
            print("A: Goodbye")
            exit()
        else:
            print("A: ")


if __name__ == "__main__":
    loop_QaBot()
