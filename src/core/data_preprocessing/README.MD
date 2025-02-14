# Data Preprocessing Tool

A Python-based tool for preprocessing keypoints stored in POSE objects.

## Overview

This tool:
- Compute mean and standard deviation for each keypoint across all POSE objects
- Extracts header object from a random POSE object, containing all extracted keypoints in a dict-like structure, separated for each pose component (i.e. head, hand, etc) 
- Stores the computed mean and standard deviation for each keypoint in pose_normalization.json, following the header objects structure
- Computes the normalized version of all POSE objects using the computed mean and standard deviation
- Stores the normalized version in a ziped file. 
