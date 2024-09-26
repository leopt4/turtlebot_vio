import math

def normalize_angle(angle):
        """
        Normalize an angle to the range between -π and π.

        Args:
            angle (float): The angle to be normalized, in radians.

        Returns:
            float: The normalized angle.
        """
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle