# Library Import
library(tidyverse)
library(lme4)
library(ggplot2)
library(lmerTest)
library(performance)

################################ Data Processing ################################

Dat_tam <- read.csv("~/Desktop/Tal/2024-10-27 Tam 2016 Data.csv")
Dat_tam$Day <- as.numeric(gsub("Day", "", Dat_tam$Time))
# Remove any NA values
Dat_tam <- Dat_tam %>% 
  filter(!is.na(Binding), !is.na(Day), Binding != "-")
# Convert Binding to numeric if it's not already
Dat_tam$Binding <- as.numeric(as.character(Dat_tam$Binding))



#################################################################################
                          # Method 1: Linear Model #
#################################################################################


################################ Model Construct ################################
models_inform <- Dat_tam %>%
  group_by(Vaccine.Scheme) %>%
  summarise(
    n = n(),
    rsq = summary(lm(Binding ~ Day))$r.squared,
    slope = coef(lm(Binding ~ Day))[2],
    intercept = coef(lm(Binding ~ Day))[1]
  )

print(models_inform)


# Statistics Summary
summary_stats <- Dat_tam %>%
  group_by(Vaccine.Scheme) %>%
  summarise(
    mean_binding = mean(Binding),
    sd_binding = sd(Binding),
    min_binding = min(Binding),
    max_binding = max(Binding),
    n_observations = n()
  )

print(summary_stats)


# List of Model
models <- Dat_tam %>%
  group_by(Vaccine.Scheme) %>%
  summarise(
    model = list(lm(Binding ~ Day))  
  )


for(i in 1:nrow(models)) {
  cat("\nModel for", models$Vaccine.Scheme[i], ":\n")
  print(summary(models$model[[i]]))
}

################################ Prediction for each Group ################################
Dat_tam$predicted <- NA 

for(scheme in unique(Dat_tam$Vaccine.Scheme)) {
  model <- models$model[[which(models$Vaccine.Scheme == scheme)]]
  
  scheme_idx <- Dat_tam$Vaccine.Scheme == scheme
  Dat_tam$predicted[scheme_idx] <- predict(model, newdata = Dat_tam[scheme_idx, ])
}

ggplot(Dat_tam, aes(x = Day, color = Vaccine.Scheme)) +
  geom_point(aes(y = Binding), alpha = 0.5) +
  geom_line(aes(y = predicted), size = 1) +
  # Customize the plot
  theme_minimal() +
  labs(title = "Measured vs Predicted Antibody Response",
       x = "Days post vaccination",
       y = "Antibody binding response",
       color = "Vaccine Scheme") +
  theme(legend.position = "bottom",
        legend.text = element_text(size = 8)) +
  facet_wrap(~Vaccine.Scheme, scales = "free_y") +
  labs(subtitle = "Points: Measured values, Lines: Model predictions")



rsquared_by_scheme <- Dat_tam %>%
  group_by(Vaccine.Scheme) %>%
  summarise(
    R2 = cor(Binding, predicted)^2,
    RMSE = sqrt(mean((Binding - predicted)^2))
  )

print(rsquared_by_scheme)


# Separate Plot(reproduce plot)
mean_values <- Dat_tam %>%
  filter(Vaccine.Scheme %in% c("Exp-inc", "Exp-dec", "Constant", "Bolus")) %>%
  group_by(Vaccine.Scheme, Day) %>%
  summarise(
    mean_binding = mean(Binding),
    se = sd(Binding)/sqrt(n())
  )

predicted_values <- Dat_tam %>%
  filter(Vaccine.Scheme %in% c("Exp-inc", "Exp-dec", "Constant", "Bolus")) %>%
  group_by(Vaccine.Scheme, Day) %>%
  do({
    model <- models$model[[which(models$Vaccine.Scheme == .$Vaccine.Scheme[1])]]
    data.frame(
      .,
      predicted = predict(model)
    )
  }) %>%
  group_by(Vaccine.Scheme, Day) %>%
  summarise(
    mean_predicted = mean(predicted)
  )

plot_data <- mean_values %>%
  left_join(predicted_values, by = c("Vaccine.Scheme", "Day"))

# Plot
ggplot(plot_data, aes(x = Day)) +
  geom_line(aes(y = mean_binding, color = Vaccine.Scheme, linetype = "Measured"), size = 1) +
  geom_line(aes(y = mean_predicted, color = Vaccine.Scheme, linetype = "Predicted"), size = 1) +
  geom_errorbar(aes(ymin = mean_binding - se, ymax = mean_binding + se, color = Vaccine.Scheme),
                width = 1, alpha = 0.3, show.legend = FALSE) +
  scale_linetype_manual(
    name = "Response Type",
    values = c("Measured" = "solid", "Predicted" = "dashed")
  ) +
  scale_color_brewer(
    palette = "Set1",
    name = "Vaccine Scheme"
  ) +
  theme_minimal() +
  labs(
    title = "Measured vs Predicted Antibody Response",
    x = "Days post vaccination",
    y = "Antibody binding response"
  ) +
  theme(
    legend.position = "bottom",
    legend.box = "vertical",
    text = element_text(size = 12)
  )


# Calculate RMSE
rmse_by_scheme <- Dat_tam %>%
  group_by(Vaccine.Scheme) %>%
  summarise(
    RMSE = sqrt(mean((Binding - predicted)^2))
  )

print(rmse_by_scheme)

#################################################################################
                        # Method 2: Random Effect Model #
#################################################################################

################################ Model Construct ################################
schemes <- unique(Dat_tam$Vaccine.Scheme)
models_list <- list()
summaries_list <- list()

for(scheme in schemes) {
  scheme_data <- Dat_tam %>%
    filter(Vaccine.Scheme == scheme)
  
  model <- lmer(Binding ~ Day + (1|Subject), data = scheme_data)
  
  models_list[[scheme]] <- model
  summaries_list[[scheme]] <- summary(model)
  
  cat("\n\n=== Model for", scheme, "===\n")
  print(summary(model))
  
  cat("\nICC for", scheme, ":\n")
  print(performance::icc(model))
}


# Prediction Plot
plot_data <- data.frame()
for(scheme in schemes) {
  scheme_data <- Dat_tam %>%
    filter(Vaccine.Scheme == scheme)
  
  scheme_data$predicted_mixed_effect <- predict(models_list[[scheme]])
  
  plot_data <- rbind(plot_data, scheme_data)
}

ggplot(plot_data, aes(x = Day, color = Vaccine.Scheme)) +
  geom_point(aes(y = Binding), alpha = 0.3) +
  geom_line(aes(y = predicted_mixed_effect), size = 1) +
  facet_wrap(~Vaccine.Scheme, scales = "free_y") +
  theme_minimal() +
  labs(title = "Mixed Effects Model Predictions by Vaccine Scheme",
       x = "Days post vaccination",
       y = "Antibody binding response")


# Use Mean Measured Value to Visualize
#plot_data_avg <- plot_data %>%
# group_by(Vaccine.Scheme, Day) %>%
#  summarise(
#    mean_binding = mean(Binding),
#    predicted = mean(predicted),
#    se = sd(Binding)/sqrt(n())  
#  )

# Use mean prediction to Visualize
#ggplot(plot_data_avg, aes(x = Day, color = Vaccine.Scheme)) +
#  geom_errorbar(aes(ymin = mean_binding - se, 
#                    ymax = mean_binding + se), 
#                alpha = 0.2, width = 1) +
#  geom_point(aes(y = mean_binding), alpha = 0.5) +
#  geom_line(aes(y = predicted), size = 1) +
#  facet_wrap(~Vaccine.Scheme, scales = "free_y") +
#  theme_minimal() +
#  labs(title = "Mixed Effects Model Predictions by Vaccine Scheme",
#       x = "Days post vaccination",
#      y = "Antibody binding response") +
#  theme(strip.text = element_text(size = 10),
#        axis.title = element_text(size = 12))

base_schemes <- c("Exp-inc", "Exp-dec", "Constant", "Bolus")

# calculate Measured Mean 
mean_bindings <- plot_data %>% 
  filter(Vaccine.Scheme %in% base_schemes) %>%
  group_by(Vaccine.Scheme, Day) %>%
  summarise(mean_binding = mean(Binding))

ggplot() +
  # Fixed Effect Model Predictions
  geom_line(data = plot_data %>% filter(Vaccine.Scheme %in% base_schemes),
            aes(x = Day, y = predicted_mixed_effect, color = Vaccine.Scheme),
            linetype = "dashed", size = 1) +
  # Observed Mean Binding
  geom_line(data = mean_bindings,
            aes(x = Day, y = mean_binding, color = Vaccine.Scheme),
            size = 1.2) +
  # Observed Individual Data Points
  geom_point(data = plot_data %>% filter(Vaccine.Scheme %in% base_schemes),
             aes(x = Day, y = Binding, color = Vaccine.Scheme),
             alpha = 0.4, size = 1.5) +
  scale_color_brewer(palette = "Set1") +
  theme_minimal() +
  labs(title = "Comparison of Fixed Effect Model Predictions and Observed Data",
       x = "Days post vaccination",
       y = "Antibody binding response") +
  theme(
    legend.position = "bottom",
    legend.box = "horizontal",
    text = element_text(size = 12)
  ) +
  guides(
    color = guide_legend(title = "Vaccine Scheme"),
    linetype = guide_legend(title = "Model Type")
  )

# No need to compare linear model with mixed effect model since the mean effect is equal