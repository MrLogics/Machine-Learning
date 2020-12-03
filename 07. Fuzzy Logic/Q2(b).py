# Develop a fuzzy expert system to detect obesity using body mass index (BMI) calculator,
# body fat (BF) and waist circumference (WC)
# Fuzzy profiling is defined on both input and output parameter as follows:
# Body Mass Index: {low, moderate, high},
# Body Fat: {low, normal, high},
# Waist Circumference: {small, medium, large} and
# Obesity: {healthy, overweight, obese}.
# The input parameters are in the range, 15kg/m2 – 35kg/m2, 15% - 35% and 30cm – 120cm,
# for body mass index (BMI), body fat (BF) and waist circumference (WC) respectively.
# Use range 1-100 for the Obesity.
# The linguistic expressions for all the input are evaluated using triangular membership
# function.
# Rule 1 If BMI=LOW and BF=LOW and WC=SMALL then OBESITY=HEALTHY
# Rule 2 If BMI= LOW and BF=LOW and WC= MEDIUM then OBESITY=HEALTHY
# Rule 3 If BMI= MODERATE and BF= NORMAL and WC= SMALL then OBESITY=OVERWEIGHT
# Rule 4 If BMI= MODERATE and BF= NORMAL and WC= MEDIUM then OBESITY=OVERWEIGHT
# Rule 5 If BMI= MODERATE and BF= NORMAL and WC= LARGE then OBESITY=OBESE
# Rule 6 If BMI= HIGH and BF= HIGH and WC= MEDIUM then OBESITY=OBESE
# Rule 7 If BMI=HIGH and BF= HIGH and WC=LARGE then OBESITY=OBESE
# a. Use triangular function for the Obesity
# b. Use the gaussian function for the Obesity



import numpy as np
import statistics
import skfuzzy as fuzz
from skfuzzy import control as ctrl 

BMI = ctrl.Antecedent(np.arange(15, 36, 1), 'BMI')
BF = ctrl.Antecedent(np.arange(15, 36, 1), 'BF')
WC = ctrl.Antecedent(np.arange(30, 121, 1), 'WC')
OBESITY = ctrl.Consequent(np.arange(1, 101, 1), 'OBESITY')

BMI['LOW'] = fuzz.trimf(BMI.universe, [15, 15, 25])
BMI['MODERATE'] = fuzz.trimf(BMI.universe, [15, 25, 35])
BMI['HIGH'] = fuzz.trimf(BMI.universe, [25, 35, 35])

BF['LOW'] = fuzz.trimf(BF.universe, [15, 15, 25])
BF['NORMAL'] = fuzz.trimf(BF.universe, [15, 25, 35])
BF['HIGH'] = fuzz.trimf(BF.universe, [25, 35, 35])


WC['SMALL'] = fuzz.trimf(WC.universe, [30, 30, 75])
WC['MEDIUM'] = fuzz.trimf(WC.universe, [30, 75, 120])
WC['LARGE'] = fuzz.trimf(WC.universe, [75, 75, 120])


OBESITY['HEALTHY'] = fuzz.gaussmf(OBESITY.universe, 0, 10)
OBESITY['OVERWEIGHT'] = fuzz.gaussmf(OBESITY.universe, 50, 15)
OBESITY['OBESE'] = fuzz.gaussmf(OBESITY.universe, 100, 10)

BMI.view()
BF.view()
WC.view()
OBESITY.view()

Rule1 = ctrl.Rule(BMI['LOW'] | BF['LOW'] | WC['SMALL'], OBESITY['HEALTHY'])
Rule2 = ctrl.Rule(BMI['LOW'] | BF['LOW'] | WC['MEDIUM'], OBESITY['HEALTHY'])
Rule3 = ctrl.Rule(BMI['MODERATE'] | BF['NORMAL'] | WC['SMALL'], OBESITY['OVERWEIGHT'])
Rule4 = ctrl.Rule(BMI['MODERATE'] | BF['NORMAL'] | WC['MEDIUM'], OBESITY['OVERWEIGHT'])
Rule5 = ctrl.Rule(BMI['MODERATE'] | BF['NORMAL'] | WC['LARGE'], OBESITY['OBESE'])
Rule6 = ctrl.Rule(BMI['HIGH'] | BF['HIGH'] | WC['MEDIUM'], OBESITY['OBESE'])
Rule7 = ctrl.Rule(BMI['HIGH'] | BF['HIGH'] | WC['LARGE'], OBESITY['OBESE'])
Rule1.view()

obesiting_ctrl = ctrl.ControlSystem([Rule1, Rule2, Rule3, Rule4, Rule5, Rule6, Rule7])
obesiting = ctrl.ControlSystemSimulation(obesiting_ctrl)

obesiting.input['BMI'] = 20
obesiting.input['BF'] = 40
obesiting.input['WC'] = 80

obesiting.compute()
print(obesiting.output['OBESITY'])
OBESITY.view(sim = obesiting)
