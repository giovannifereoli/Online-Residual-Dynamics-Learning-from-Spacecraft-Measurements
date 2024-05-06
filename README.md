<p align="center">
  <img align="center" src="https://www.colorado.edu/brand/sites/default/files/styles/medium/public/page/boulder-one-line-reverse.png?itok=jWuueUXe" width="400" />
</p>

<div align="center">
  
![GitHub last commit](https://img.shields.io/github/last-commit/giovannifereoli/Online-Residual-Dynamics-Learning-from-Spacecraft-Measurements)
[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.0.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

</div>

## Online Residual Dynamics Learning from Spacecraft Measurements 

The utilization of Artificial Neural Networks (ANNs) to approximate spacecraft dynamics offers a means to improve the accuracy and adaptability of the onboard model, thereby empowering the spacecraft with greater autonomy. Traditional analytical models, often based on linearization and devoid of perturbation modeling due to complexity or lack of knowledge, reach their limitations. Considering future space endeavors involving celestial bodies such as asteroids, characterized by uncertain gravitational perturbations and unknown physical influences, or scenarios such as Entry, Descent, and Landing (EDL) with imprecise aerodynamic models, or even quadrotor flights with uncertainties during high-speed maneuvering, this paper proposes a hybrid approach. This approach integrates first-principles models with an Online Supervised Training (OST) method to capture disturbance and residual dynamics. OST strives to mitigate the impact of modeling errors, external disturbances, and non-linearities. Pseudo-labels for training are generated using an Extended Kalman filter, enabling the neural network to learn from spacecraft measurements and incorporate residual dynamics in real-time. The results of this research, due to the differentiability of neural networks, promise improved performance in navigation and control algorithms with unparalleled accuracy and robustness. In addition, it offers a means to expedite mission phases dedicated to constructing more precise dynamical models, thereby assisting engineers and scientists in mission planning and execution.

## Credits
This project has been created by [Giovanni Fereoli](https://github.com/giovannifereoli) in 2024.
For any problem, clarification or suggestion, you can contact the author at [giovanni.fereoli@colorado.edu](mailto:giovanni.fereoli@colorado.edu).

## License
The package is under the [MIT](https://choosealicense.com/licenses/mit/) license.
