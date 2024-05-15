# Exploring Pulsar Signals using HTRU Dataset: Data Analysis, Classification, and Visualization
## HTRU-Pulsar-Star-Dataset
This project focuses on analyzing pulsar data, conducting exploratory data analysis, training machine learning models for classification, and visualizing results.

# High Time Resolution Universe (HTRU): All-sky survey for pulsars and radio transients at a frequency of 1400 MHz

# HTRU-I Dataset

### Abstract: 
Pulsar candidates collected during the HTRU survey. Pulsars are a type of star, of considerable scientific interest. Candidates must be classified in to pulsar and non-pulsar classes to aid discovery.

### Motivation
Pulsar stars are a very rare type of Neutron star that produce radio emission detectable on Earth and they are of considerable scientific interest as probes of space-time and states of matter. Their emission spreads across the sky and produces a detectable pattern of broadband radio emission. However in practice almost all detections are caused by radio frequency interference and noise, making legitimate signals hard to find.

Having said that, the main purpose of this problem is to build a simple classifier using deep learning tools in order to predict wether a detected signal comes from a pulsar star or from other sources such as noises, interferences, etc.

###### High Time Resolution Universe - 
The High Time Resolution Universe (HTRU) is an all-sky survey for pulsars and radio transients at a frequency of 1400 MHz.

The Southern Hemisphere is being observed with the Parkes Multi-Beam system, the Northern Hemisphere is being observed with the Effelsberg 7-beam system. It is expected that the survey sensitivity will be similar for  both hemispheres.

These surveys will have a much higher frequency and time resolution than previous surveys like the Parkes Multi-Beam Survey. Because of this, they will suffer much less from dispersive smearing and will therefore be able to detect a much larger number of millisecond pulsars, particularly near the Galactic plane, where we expect the most exciting binary systems and a much larger population of pulsars. These surveys will likely find hundreds of millisecond pulsars, which are great laboratories for the study of fundamental physics, gravitational astronomy and astrophysics in general. Furthermore, the Effelsberg part of the survey (HTRU-North) will survey the whole Northern Hemisphere for the first time in 30 years,allowing a new high-sensitive view onto this part of the sky. The Northern Hemisphere surveys have a great advantage that new discoveries can be followed up with a large variety of radio telescopes, which maximizes scientific output.

###### Link: 
https://archive.ics.uci.edu/ml/datasets/HTRU2

###### Method - 2: 
https://psrsigsim.readthedocs.io/en/latest/pulse_nulling_example.html

###### Deep Learning Method: 
https://datauab.github.io/pulsar_stars/

###### Source:
Dr Robert Lyon, University of Manchester, School of Physics and Astronomy, Alan Turing Building, Manchester M13 9PL, United Kingdom, robert.lyon '@' manchester.ac.uk

#### Data Set Information:
HTRU2 is a data set which describes a sample of pulsar candidates collected during the High Time Resolution Universe Survey (South) [1].

Pulsars are a rare type of Neutron star that produce radio emission detectable here on Earth. They are of considerable scientific interest as probes of space-time, the inter-stellar medium, and states of matter (see [2] for more uses).

As pulsars rotate, their emission beam sweeps across the sky, and when this crosses our line of sight, produces a detectable pattern of broadband radio emission. As pulsars
rotate rapidly, this pattern repeats periodically. Thus pulsar search involves looking for periodic radio signals with large radio telescopes.

Each pulsar produces a slightly different emission pattern, which varies slightly with each rotation (see [2] for an introduction to pulsar astrophysics to find out why). Thus a potential signal detection known as a 'candidate', is averaged over many rotations of the pulsar, as determined by the length of an observation. In the absence of additional info, each candidate could potentially describe a real pulsar. However in practice almost all detections are caused by radio frequency interference (RFI) and noise, making legitimate signals hard to find.

Machine learning tools are now being used to automatically label pulsar candidates to facilitate rapid analysis. Classification systems in particular are being widely adopted,
(see [4,5,6,7,8,9]) which treat the candidate data sets as binary classification problems. Here the legitimate pulsar examples are a minority positive class, and spurious examples the majority negative class. At present multi-class labels are unavailable, given the costs associated with data annotation.

The data set shared here contains 16,259 spurious examples caused by RFI/noise, and 1,639 real pulsar examples. These examples have all been checked by human annotators.

The data is presented in two formats: CSV and ARFF (used by the WEKA data mining tool). Candidates are stored in both files in separate rows. Each row lists the variables first, and the class label is the final entry. The class labels used are 0 (negative) and 1 (positive).

Please note that the data contains no positional information or other astronomical details. It is simply feature data extracted from candidate files using the PulsarFeatureLab tool (see [10]).

#### Here are the terminologies corresponding to the columns in your pulsar star dataset:

1. **Mean of the integrated profile**: This refers to the average intensity of the integrated pulse profile of a pulsar. The integrated profile represents the combined signal intensity across different phases of the pulsar's rotation.

2. **Standard deviation of the integrated profile**: This measures the variability or spread of the integrated pulse profile intensities around the mean value. A higher standard deviation indicates greater variability in the profile's intensity.

3. **Excess kurtosis of the integrated profile**: Kurtosis is a statistical measure that describes the shape of a distribution's tails relative to its peak. Excess kurtosis quantifies how much the distribution deviates from a normal distribution in terms of its tail heaviness.

4. **Skewness of the integrated profile**: Skewness measures the asymmetry of a distribution. Positive skewness indicates a longer tail on the right side of the distribution, while negative skewness indicates a longer tail on the left side.

5. **Mean of the DM-SNR curve**: This represents the average signal-to-noise ratio (SNR) of the dynamic spectrum of the pulsar's emission. The DM-SNR curve reflects the dispersion and intensity variations of the pulsar signal across different frequencies.

6. **Standard deviation of the DM-SNR curve**: Similar to the standard deviation of the integrated profile, this parameter measures the variability or dispersion of the SNR values in the DM-SNR curve.

7. **Excess kurtosis of the DM-SNR curve**: Excess kurtosis of the DM-SNR curve quantifies the shape of the SNR distribution in the dynamic spectrum, indicating how heavy or light the tails of the distribution are compared to a normal distribution.

8. **Skewness of the DM-SNR curve**: Skewness of the DM-SNR curve measures the asymmetry of the SNR distribution in the dynamic spectrum. Positive skewness indicates asymmetry with a longer tail on the right side, while negative skewness indicates asymmetry with a longer tail on the left side.

9. **Class**: This column likely represents the classification or label of each data point, indicating whether it corresponds to a pulsar (positive class) or a non-pulsar (negative class).

These terminologies describe the statistical properties, distribution shapes, and signal characteristics captured by the columns in your dataset, providing valuable insights into the nature of pulsar stars and their observational data.

#### Citation Request:

If you use the dataset in your work, please cite us using the following paper:

R. J. Lyon, B. W. Stappers, S. Cooper, J. M. Brooke, J. D. Knowles, Fifty Years of Pulsar Candidate Selection: From simple filters to a new principled real-time classification approach, Monthly Notices of the Royal Astronomical Society 459 (1), 1104-1123, DOI: 10.1093/mnras/stw656

If possible, please also cite the DOI of the data set directly:
R. J. Lyon, HTRU2, DOI: 10.6084/m9.figshare.3080389.v1.

**Acknowledgements**

This data was obtained with the support of grant EP/I028099/1 for the University of Manchester Centre for Doctoral Training in Computer Science, from the UK Engineering and Physical Sciences Research Council (EPSRC). The raw observational data was collected by the High Time Resolution Universe Collaboration using the Parkes Observatory, funded by the Commonwealth of Australia and managed by the CSIRO.

### More Details:
https://www.mpifr-bonn.mpg.de/research/fundamental/htru
