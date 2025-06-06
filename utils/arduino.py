import serial
import time

# =======================================
# Configuration
# =======================================
SERIAL_PORT = "/dev/tty.usbserial-0001"     # e.g., 'COM5' on Windows or '/dev/ttyACM0' on Linux/macOS
BAUD_RATE   = 115200     # Must match your Arduino sketch

# We'll keep a global reference to the Serial object
arduino_serial = None


# =======================================
# Connection Functions
# =======================================
def initialize_connection(port=SERIAL_PORT, baud=BAUD_RATE):
    """
    Initializes and returns a serial connection to the Arduino.
    Call this once when your application/server starts.
    """
    global arduino_serial
    try:
        arduino_serial = serial.Serial(port, baud, timeout=1)
        # Give Arduino time to reset
        time.sleep(2)
        print(f"[INFO] Connected to Arduino on {port} at {baud} baud.")
    except serial.SerialException as e:
        raise RuntimeError(f"Failed to connect to Arduino on port {port}: {e}")


def close_connection():
    """
    Closes the serial connection to the Arduino.
    Call this when your application/server stops.
    """
    global arduino_serial
    if arduino_serial and arduino_serial.is_open:
        arduino_serial.close()
        arduino_serial = None
        print("[INFO] Arduino serial connection closed.")


def send_command(command):
    """
    Sends a single line (text command) to the Arduino.
    Make sure your Arduino sketch is set up to parse these commands.
    """
    global arduino_serial
    if not arduino_serial or not arduino_serial.is_open:
        raise RuntimeError("Serial connection not open. Call initialize_connection() first.")

    line = (command + "\n").encode("utf-8")
    arduino_serial.write(line)
    arduino_serial.flush()
    # Optionally read a response:
    # response = arduino_serial.readline().decode('utf-8').strip()
    # print(f"[Arduino] {response}")


# =======================================
# 1) Move Servo 0° → 100°
# =======================================
def move_servo_0_to_100():
    """
    Moves the servo from (or to) 0°, waits 1 second, then moves to 100°, and stops.
    """
    # Tell Arduino: set servo to 0°
    send_command("RIGHT")  
    time.sleep(1)           # Wait 1 second
    send_command("RIGHT")
    time.sleep(1)           # Wait 1 second
    send_command("RIGHT")    
    # No extra waiting needed unless you want to hold the servo at 100° for some time
    # "Stop" just means we stop sending more servo commands.


# =======================================
# 2) Move Servo 100° → 0°
# =======================================
def move_servo_100_to_0():
    """
    Moves the servo from (or to) 100°, waits 1 second, then moves to 0°, and stops.
    """
    send_command("LEFT")  
    time.sleep(1)           # Wait 1 second
    send_command("LEFT")
    time.sleep(1)           # Wait 1 second
    send_command("LEFT") 
    # Again, no extra waiting needed to "stop."


# =======================================
# 3) Start Motors (Forward Slow)
# =======================================
def start_motors_slow():
    """
    Commands the Arduino to set the left and right motors to a slow forward speed.
    For example, this might correspond to direction=HIGH, PWM=55, etc.
    """
    send_command("MOTORS_FORWARD_SLOW")


# =======================================
# 4) Stop Motors
# =======================================
def stop_motors():
    """
    Commands the Arduino to stop both motors immediately (PWM=0).
    """
    send_command("MOTORS_STOP")
