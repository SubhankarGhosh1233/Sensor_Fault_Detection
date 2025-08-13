
# Sensor APS (Air Pressure System) Fault Detection
**Problem Statement:**

The Air Pressure System (APS) is a critical component of a heavy-duty vehicle that uses compressed air to force a piston to provide pressure to the brake pads, slowing the vehicle down. The benefits of using an APS instead of a hydraulic system are the easy availability and long-term sustainability of natural air.

This is a Binary Classification problem, in which the affirmative class indicates that the failure was caused by a certain component of the APS, while the negative class indicates that the failure was caused by something else.

**Solution Proposed:**

In this project, the system in focus is the Air Pressure system (APS) which generates pressurized air that are utilized in various functions in a truck, such as braking and gear changes. The datasets positive class corresponds to component failures for a specific component of the APS system. The negative class corresponds to trucks with failures for components not related to the APS system.

The problem is to reduce the cost due to unnecessary repairs. So it is required to minimize the false predictions.

## Deployment

**Platform:** Render

**Server:** gunicorn with gevent workers for async performance

**url:** https://aps-sensor-fault-detection.onrender.com


## Sample File for Testing
Due to memory limitations on Render’s free tier, please use the small sample CSV provided in this repo for testing the upload functionality.

**File Location:**
Sample_File_for_Testing\aps_failure_training_set1.csv

# Tech Stack
1. Python
2. Machine learning algorithms
3. MongoDB
4. Flask (for API and web interface)