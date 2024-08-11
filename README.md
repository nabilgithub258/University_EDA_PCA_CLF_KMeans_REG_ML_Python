---

# University Dataset Analysis

## Project Overview

This project involves analyzing a university dataset to predict key outcomes using both classification and regression models. Our primary objectives were to classify universities as private or public and to predict graduation rates based on various university attributes.

## Models and Techniques Used

### Classification
- **Model**: GradientBoostingClassifier
- **Performance**: Achieved 91% accuracy and precision.
- **Additional Techniques**: Applied SMOTE to address class imbalance.

### Regression
- **Model**: Linear Regression
- **Target Variable**: Graduation Rate
- **Performance**: RMSE of 12 on a scale of 0-100.
- **Data Integrity**: Excluded `private_uni` variable to prevent data leakage.

## Dataset

The dataset includes various features related to universities, such as application numbers, enrollment statistics, faculty qualifications, financial costs, and more. Key columns include:

- **Private**: Indicates if the university is private or public.
- **Grad.Rate**: Graduation rate of students.
- **Apps, Accept, Enroll**: Number of applications, acceptances, and enrollments.
- **Outstate**: Tuition for out-of-state students.
- **PhD**: Percentage of faculty with a Ph.D.

## Results

- **Classification**: Successfully classified universities with high accuracy, leveraging GradientBoostingClassifier.
- **Regression**: Predicted graduation rates with an RMSE of 12, balancing model complexity and predictive power.

## Conclusion

This project demonstrates the application of machine learning techniques to predict university-related metrics, providing valuable insights despite the challenges posed by a small, imbalanced dataset.

## Contact

For any inquiries or feedback, feel free to reach out to [nabilmomin1989@gmail.com].

---
