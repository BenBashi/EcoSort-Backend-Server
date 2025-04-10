# 1. system_start(): 
#    - Activates the treadmill using start_treadmill().
#    - Initiates the image_scan() process, which:
#        * captures frames from a USB camera every 2 seconds,
#        * classifies each frame (classify_image()),
#        * directs trash to the appropriate bin using rotate_arm_right() or rotate_arm_left(),
#        * or if "Other", allows trash to go to the main bin,
#        * or if "Failure", triggers system_stop(),
#        * updates the database via create_sample(),
#        * repeats until stopped by the user or a failure occurs.

def system_start():
    """
    Skeleton for starting the system:
      1) Activate treadmill.
      2) Begin scanning loop.
         - Capture images.
         - Classify images.
         - Direct to bin or fail.
         - Update database.
    """
    # 1) Activate treadmill
    # TODO: Implement start_treadmill()
    # start_treadmill()
    
    # 2) Initiate image_scan() loop
    #     You can run it in a separate thread or simply call a function with a while-loop.
    #     For example:
    # image_scan()

    pass


# 2. system_stop():
#    - Stop the treadmill using stop_treadmill().
#    - Terminates the image_scan() loop.

def system_stop():
    """
    Skeleton for stopping the system:
      - Stop the treadmill,
      - Break out of the scanning loop.
    """
    # 1) Deactivate/stop treadmill
    # TODO: Implement stop_treadmill()
    # stop_treadmill()
    
    # 2) Terminate the scanning loop
    #     Possibly set a global or shared variable that tells image_scan() to exit:
    # is_running = False
    pass
