"""
oak_pipeline.app
~~~~~~~

Main application.

:author: Mats Fockaert
:copyright: # TODO
:license: # TODO
"""

from controller import Controller

import sys



def main():
    """The main function of the application.
    
    This function initializes the Controller, sets up the cameras based on the config file,
    checks if a GUI option is provided, and then retrieves and prints the detections.
    """
    controller = Controller()
    config = "./config.json"
    controller.make_cameras(config)
    gui = False
    # Check if an argument is provided
    if len(sys.argv) > 1:
        gui_option = sys.argv[1]

        if gui_option == "gui-1":
            gui = True

    detections = controller.get_detections(gui)
    print(detections)



if __name__ == "__main__":
    main()

