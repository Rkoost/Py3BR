from scipy import constants

K2HAR = 1/constants.physical_constants['hartree-kelvin relationship'][0]
BOH2M = constants.physical_constants['Bohr radius'][0]
ANG2BOH = 1/(BOH2M*1E10)
U2ME = 1/constants.physical_constants['electron mass in u'][0]
J2HAR = constants.physical_constants['joule-hartree relationship'][0]
HAR2INVCM = constants.physical_constants['hartree-inverse meter relationship'][0]/100
T2S = constants.physical_constants['atomic unit of time'][0]
AU2CM = constants.physical_constants['Bohr radius'][0]*100