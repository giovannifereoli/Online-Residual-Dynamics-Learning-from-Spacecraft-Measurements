<p align="center">
  <img align="center" src="https://www.colorado.edu/brand/sites/default/files/styles/medium/public/page/boulder-one-line-reverse.png?itok=jWuueUXe" width="400" />
</p>

<div align="center">
  
![GitHub last commit](https://img.shields.io/github/last-commit/giovannifereoli/Online-Residual-Dynamics-Learning-from-Spacecraft-Measurements)
[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.0.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

</div>

## Online Residual Dynamics Learning from Spacecraft Measurements 

Employing Artificial Neural Networks (ANNs) to learn spacecraft dynamical disturbances enhances the precision and adaptability of the models on-board, thus granting the spacecraft greater autonomy. Conventional analytical models rooted in first-principles, often reliant on linearization and vulnerable to modeling errors, as well as neglecting complex disturbances due to limited knowledge, face inherent limitations. Looking ahead to future space missions involving celestial bodies such as asteroids, characterized by uncertain gravitational fields, or scenarios such as Entry, Descent, and Landing (EDL) with imprecise aerodynamic models, this paper advocates for a hybrid dynamical modeling approach. This strategy integrates first-principles models with an Online Supervised Training (OST) method to capture residual dynamics. Utilizing Extended Kalman Filter (EKF) estimates with Dynamical Model Compensation (DMC) generates features and labels for training, enabling the neural network to learn from spacecraft measurements and adapt to residual dynamics in real-time. The results of this study, bolstered by the differentiability of neural networks, promise enhanced performance in Guidance, Navigation, and Control (GNC) algorithms with unprecedented accuracy and robustness. Moreover, it expedites mission phases dedicated to refining dynamical models, thus aiding engineers and scientists in mission planning and execution.

## Credits
This project has been created by [Giovanni Fereoli](https://github.com/giovannifereoli) in 2024.
For any problem, clarification or suggestion, you can contact the author at [giovanni.fereoli@colorado.edu](mailto:giovanni.fereoli@colorado.edu).

## License
The package is under the [MIT](https://choosealicense.com/licenses/mit/) license.
