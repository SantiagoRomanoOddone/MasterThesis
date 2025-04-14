# 📊 Master Thesis: Forecasting Demand Across Products and Stores

## 🧠 Overview

This project is the **final thesis for my Master’s in Management and Analytics**, combining academic research with applied machine learning to tackle a highly relevant problem in the retail industry: **demand forecasting at scale**.

In the dynamic world of retail, accurately forecasting demand is pivotal. It ensures that the right products are available at the right time and place, optimizing inventory levels and enhancing customer satisfaction. This project addresses the challenge of forecasting demand across a vast array of product-store combinations using **cutting-edge deep learning models**, pushing the boundaries of what’s possible in retail forecasting.

## 🚀 Project Objectives

- **Explore and build powerful deep learning models** to generate highly accurate and robust demand forecasts.
- **Investigate** how modern architectures can learn from vast and diverse time series data, revealing what makes forecasting easier—or harder—in the retail context.
- **Implement and evaluate** advanced models across thousands of product-store combinations, identifying patterns, uncovering challenges, and benchmarking performance.
- **Leverage probabilistic forecasting** to capture uncertainty and improve decision-making in real-world retail operations.

This is both a **technical exploration and hands-on implementation**, grounded in business impact and backed by real-world data complexity.

## 🧰 Methodology

To meet these goals, this project uses the **GluonTS** library—a powerful toolkit for probabilistic time series modeling. GluonTS makes it easy to experiment with and compare several state-of-the-art models, including:

- **DeepAR**: An autoregressive recurrent network for capturing complex temporal dependencies.
- **Temporal Fusion Transformer (TFT)**: Combines LSTM layers and attention for interpretable multi-horizon forecasting.
- **WaveNet**: A dilated convolutional model that captures long-range dependencies and can predict in parallel.
- **Simple Feedforward (MLP)**: A baseline fully connected network that provides a reference point for more complex models.

By standardizing the modeling pipeline, GluonTS enables rapid experimentation, consistent evaluation, and flexible integration of both static and dynamic features.

## 🛒 Relevance to the Retail Industry

Demand forecasting is not just a technical challenge—it’s a **mission-critical function**. Retailers depend on precise forecasts to:

- Reduce stockouts and overstocking
- Optimize logistics and supply chain decisions
- Improve promotional and pricing strategies
- Align business planning with market realities

**Deep learning models**, when applied correctly, offer significant advantages:

- **Probabilistic output** helps businesses plan for a range of possible outcomes.
- **Scalability** makes it feasible to forecast for thousands of product-store combinations in parallel.
- **Feature integration** allows the inclusion of rich covariates like seasonality, holidays, weather, and promotions.

This project demonstrates how advanced AI can deliver real-world value—and how **academic innovation** can inform better business strategy.

## 📈 Key Outcomes Achieved

- 🚀 **Deep learning models outperformed state-of-the-art baselines**, including widely used algorithms like XGBoost and CatBoost, delivering superior accuracy across diverse forecasting scenarios.
- ⚙️ **Scalable and robust implementations** successfully predicted demand for thousands of product-store combinations simultaneously.
- 🔍 **Extracted actionable insights** on key demand drivers (e.g., promotions, holidays), helping to explain when and why deep learning models excel.
- 🧠 Developed a **powerful forecasting framework** using GluonTS, easily adaptable to other domains facing complex multi-series prediction challenges.

---

*This project blends deep learning, experimentation, and applied research to deliver real impact in one of the most important areas of retail: understanding and anticipating demand.*
