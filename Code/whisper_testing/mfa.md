To run MFA do:
If no current dedicated ENV: 
- conda create -n aligner -c conda-forge python=3.10 montreal-forced-aligner
Else - ignore step above

- conda activate aligner
- mfa model download dictionary french_mfa
- mfa model download acoustic french_mfa

Put the gold transcript and the Audio file in the same isolated Directory. they must have the same name --> ID.txt and ID.wav.
Used Data/MFA_input as directory
- mfa align Data/MFA_input french_mfa french_mfa Data/MFA_out

