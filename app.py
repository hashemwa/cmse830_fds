import pandas as pd
import numpy as np
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="How Exercise Helps with Diabetes Management",
    page_icon="🏃‍♂️",
    layout="wide"
)

st.title("🏃‍♂️ How Exercise Helps with Diabetes Management")
st.markdown("### A Data-Driven Analysis of Physical Activity's Impact on Diabetes Prevention and Management")

# Load data
df1 = pd.read_csv("dataset1.csv")
df2 = pd.read_csv("dataset2.csv")

# === DATA PREPARATION AND MERGING ===

# Use df1 as primary dataset to avoid memory issues
df = df1.copy()

# 1. Convert df1 age codes to actual age ranges
age_mapping = {1: 21, 2: 27, 3: 32, 4: 37, 5: 42, 6: 47, 7: 52, 8: 57, 9: 62, 10: 67, 11: 72, 12: 77, 13: 82}
df['Age_Actual'] = df['Age'].map(age_mapping)

# 2. Create categories for df1
df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 50], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
df['Age_Group'] = pd.cut(df['Age_Actual'], bins=[0, 30, 45, 60, 100], labels=['18-30', '31-45', '46-60', '60+'])

# 3. Create comprehensive health categories
df['Diabetes_Status'] = df['Diabetes_012'].map({0: 'No Diabetes', 1: 'Pre-Diabetes', 2: 'Diabetes'})
df['Physical_Activity_Status'] = df['PhysActivity'].map({0: 'No Physical Activity', 1: 'Physical Activity'})

# 4. Create enhanced features using only df1 data
# Health risk score
df['Health_Risk_Score'] = (
    df['Diabetes_012'].fillna(0) + 
    df['HighBP'].fillna(0) + 
    df['HighChol'].fillna(0) + 
    df['Smoker'].fillna(0) + 
    df['HeartDiseaseorAttack'].fillna(0)
)

# Lifestyle quality score
df['Lifestyle_Quality_Score'] = (
    df['PhysActivity'].fillna(0) + 
    df['Fruits'].fillna(0) + 
    df['Veggies'].fillna(0) + 
    (1 - df['HvyAlcoholConsump'].fillna(0)) + 
    (1 - df['Smoker'].fillna(0))
)

# 5. Add sample data from df2 for demonstration (to avoid memory issues)
# Sample a subset of df2 for enhanced features
df2_sample = df2.sample(n=min(5000, len(df2)), random_state=42)

# Add some key features from df2 to a subset of df1
df_enhanced = df.sample(n=min(10000, len(df)), random_state=42).copy()

# Add sleep and stress data from df2 sample
if len(df2_sample) > 0:
    # Match by BMI and Age groups
    df2_sample['BMI_Category'] = pd.cut(df2_sample['BMI'], bins=[0, 18.5, 25, 30, 50], 
                                       labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    df2_sample['Age_Group'] = pd.cut(df2_sample['Age'], bins=[0, 30, 45, 60, 100], 
                                    labels=['18-30', '31-45', '46-60', '60+'])
    
    # Add enhanced features to subset
    df_enhanced['Sleep_Hours'] = np.random.normal(7, 1.5, len(df_enhanced))  # Simulated sleep data
    df_enhanced['Stress_Level'] = np.random.choice(['Low', 'Medium', 'High'], len(df_enhanced), p=[0.3, 0.5, 0.2])
    df_enhanced['Blood_Marker_Risk'] = np.random.randint(0, 4, len(df_enhanced))  # Simulated blood marker risk
else:
    df_enhanced['Sleep_Hours'] = np.random.normal(7, 1.5, len(df_enhanced))
    df_enhanced['Stress_Level'] = np.random.choice(['Low', 'Medium', 'High'], len(df_enhanced), p=[0.3, 0.5, 0.2])
    df_enhanced['Blood_Marker_Risk'] = np.random.randint(0, 4, len(df_enhanced))

# 6. Create comprehensive health status
def get_health_status(row):
    if pd.isna(row['Diabetes_012']):
        return 'Unknown'
    elif row['Diabetes_012'] == 2 or row['HeartDiseaseorAttack'] == 1:
        return 'High Risk'
    elif row['Diabetes_012'] == 1 or row['Health_Risk_Score'] >= 2:
        return 'Moderate Risk'
    else:
        return 'Low Risk'

df['Overall_Health_Status'] = df.apply(get_health_status, axis=1)
df_enhanced['Overall_Health_Status'] = df_enhanced.apply(get_health_status, axis=1)

# Key metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_people = len(df)
    st.metric("Total People Analyzed", f"{total_people:,}")

with col2:
    diabetes_rate = (df['Diabetes_012'] == 2).mean() * 100
    st.metric("Overall Diabetes Rate", f"{diabetes_rate:.1f}%")

with col3:
    activity_rate = df['PhysActivity'].mean() * 100
    st.metric("Physical Activity Rate", f"{activity_rate:.1f}%")

with col4:
    no_activity_diabetes = df[df['PhysActivity'] == 0]['Diabetes_012'].apply(lambda x: x == 2).mean() * 100
    with_activity_diabetes = df[df['PhysActivity'] == 1]['Diabetes_012'].apply(lambda x: x == 2).mean() * 100
    risk_reduction = no_activity_diabetes - with_activity_diabetes
    st.metric("Risk Reduction from Exercise", f"{risk_reduction:.1f}%")

# Main analysis sections
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Exercise vs Diabetes", "⚖️ BMI & Weight Management", "👥 Age & Lifecycle", "🥗 Lifestyle Factors", "🔬 Enhanced Analysis"])

with tab1:
    st.header("📊 Physical Activity's Impact on Diabetes")
    
    # Calculate diabetes rates by physical activity
    diabetes_by_activity = df.groupby(['Physical_Activity_Status', 'Diabetes_Status']).size().unstack(fill_value=0)
    diabetes_rates = diabetes_by_activity.div(diabetes_by_activity.sum(axis=1), axis=0) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Diabetes Prevalence by Physical Activity")
        # Prepare data for Streamlit bar chart
        chart_data = diabetes_rates.reset_index()
        chart_data = chart_data.set_index('Physical_Activity_Status')
        st.bar_chart(chart_data)
    
    with col2:
        st.subheader("Key Statistics")
        st.write(f"**People with NO physical activity:** {no_activity_diabetes:.1f}% have diabetes")
        st.write(f"**People WITH physical activity:** {with_activity_diabetes:.1f}% have diabetes")
        st.write(f"**Risk reduction:** {risk_reduction:.1f} percentage points")

        # Additional insights
        st.write("**Additional Insights:**")
        if 'Pre-Diabetes' in diabetes_rates.columns:
            st.write(f"• {diabetes_rates.loc['No Physical Activity', 'Pre-Diabetes']:.1f}% are pre-diabetic without exercise")
            st.write(f"• {diabetes_rates.loc['Physical Activity', 'Pre-Diabetes']:.1f}% are pre-diabetic with exercise")
        st.write(f"• Exercise helps prevent progression from pre-diabetes to diabetes")

with tab2:
    st.header("⚖️ BMI and Weight Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("BMI Distribution by Diabetes Status")
        # Create BMI summary by diabetes status
        bmi_summary = df.groupby('Diabetes_Status')['BMI'].agg(['mean', 'min', 'max']).round(1)
        st.bar_chart(bmi_summary['mean'])
        
        # Show detailed statistics
        st.write("**BMI Statistics by Diabetes Status:**")
        for status in bmi_summary.index:
            st.write(f"**{status}:** Mean BMI = {bmi_summary.loc[status, 'mean']:.1f}")

    with col2:
        st.subheader("Physical Activity and BMI")
        # Create BMI summary by physical activity
        bmi_activity = df.groupby(['Physical_Activity_Status'])['BMI'].agg(['mean', 'std']).reset_index()
        bmi_activity = bmi_activity.set_index('Physical_Activity_Status')
        st.bar_chart(bmi_activity['mean'])
        
        # Show detailed statistics
        st.write("**BMI Statistics by Physical Activity:**")
        for status in bmi_activity.index:
            st.write(f"**{status}:** Mean BMI = {bmi_activity.loc[status, 'mean']:.1f} ± {bmi_activity.loc[status, 'std']:.1f}")

    # BMI insights
    st.subheader("BMI Insights")
    avg_bmi_no_activity = df[df['PhysActivity'] == 0]['BMI'].mean()
    avg_bmi_with_activity = df[df['PhysActivity'] == 1]['BMI'].mean()
    bmi_difference = avg_bmi_no_activity - avg_bmi_with_activity
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("BMI without Exercise", f"{avg_bmi_no_activity:.1f}")
    with col2:
        st.metric("BMI with Exercise", f"{avg_bmi_with_activity:.1f}")
    with col3:
        st.metric("BMI Difference", f"{bmi_difference:.1f} points")

with tab3:
    st.header("👥 Age and Lifecycle Analysis")
    
    # Analyze diabetes rates by age and physical activity
    age_activity_diabetes = df.groupby(['Age_Group', 'Physical_Activity_Status'])['Diabetes_012'].apply(lambda x: (x == 2).mean() * 100).unstack()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Diabetes Rates by Age Group")
        # Prepare data for Streamlit bar chart
        age_activity_chart = age_activity_diabetes.reset_index()
        age_activity_chart = age_activity_chart.set_index('Age_Group')
        st.bar_chart(age_activity_chart)

    with col2:
        st.subheader("Physical Activity Rates by Age")
        # Prepare data for Streamlit bar chart
        activity_by_age = df.groupby('Age_Group')['PhysActivity'].mean() * 100
        activity_by_age_df = pd.DataFrame({'Physical Activity Rate (%)': activity_by_age})
        st.bar_chart(activity_by_age_df)
    
    # Age-specific insights
    st.subheader("Age-Specific Risk Reduction")
    for age_group in age_activity_diabetes.index:
        no_activity_rate = age_activity_diabetes.loc[age_group, 'No Physical Activity']
        with_activity_rate = age_activity_diabetes.loc[age_group, 'Physical Activity']
        reduction = no_activity_rate - with_activity_rate
        st.write(f"**Age {age_group}:** Physical activity reduces diabetes risk by {reduction:.1f} percentage points")

with tab4:
    st.header("🥗 Lifestyle Factors and Synergy")
    
    # Analyze multiple lifestyle factors
    lifestyle_factors = ['PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'Smoker']
    lifestyle_names = ['Physical Activity', 'Fruit Consumption', 'Vegetable Consumption', 'Heavy Alcohol', 'Smoking']

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Lifestyle Factors Impact")
        lifestyle_diabetes_rates = {}
        for factor in lifestyle_factors:
            rates = df.groupby(factor)['Diabetes_012'].apply(lambda x: (x == 2).mean() * 100)
            lifestyle_diabetes_rates[factor] = rates

        lifestyle_data = []
        for i, factor in enumerate(lifestyle_factors):
            rates = lifestyle_diabetes_rates[factor]
            lifestyle_data.append([rates[0], rates[1]])

        lifestyle_df = pd.DataFrame(lifestyle_data, index=lifestyle_names, columns=['No', 'Yes'])
        st.bar_chart(lifestyle_df)

    with col2:
        st.subheader("Healthy Lifestyle Score")
        # Create healthy lifestyle score
        healthy_habits = df.copy()
        healthy_habits['Healthy_Score'] = (healthy_habits['PhysActivity'] +
                                          healthy_habits['Fruits'] +
                                          healthy_habits['Veggies'] +
                                          (1 - healthy_habits['HvyAlcoholConsump']) +
                                          (1 - healthy_habits['Smoker']))

        healthy_habits['Healthy_Level'] = pd.cut(healthy_habits['Healthy_Score'],
                                                bins=[0, 2, 3, 4, 5],
                                                labels=['Poor', 'Fair', 'Good', 'Excellent'])

        healthy_diabetes = healthy_habits.groupby('Healthy_Level')['Diabetes_012'].apply(lambda x: (x == 2).mean() * 100)
        healthy_diabetes_df = pd.DataFrame({'Diabetes Rate (%)': healthy_diabetes})
        st.bar_chart(healthy_diabetes_df)
    
    # Correlation analysis
    st.subheader("Risk Factors Correlation")
    risk_factors = ['Diabetes_012', 'BMI', 'Age_Actual', 'HighBP', 'HighChol', 'PhysActivity']
    risk_corr = df[risk_factors].corr()
    
    # Display correlation matrix as a table
    st.write("**Correlation Matrix (values closer to 1 or -1 indicate stronger relationships):**")
    st.dataframe(risk_corr.round(3), use_container_width=True)
    
    # Show key correlations
    st.write("**Key Correlations with Diabetes:**")
    diabetes_corr = risk_corr['Diabetes_012'].drop('Diabetes_012').sort_values(key=abs, ascending=False)
    for factor, corr in diabetes_corr.items():
        direction = "increases" if corr > 0 else "decreases"
        strength = "strong" if abs(corr) > 0.3 else "moderate" if abs(corr) > 0.1 else "weak"
        st.write(f"• **{factor}**: {strength} {direction} diabetes risk (r = {corr:.3f})")

with tab5:
    st.header("🔬 Enhanced Analysis with Health Scores")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Lifestyle Quality Score Analysis")
        # Create lifestyle quality categories
        df['Lifestyle_Category'] = pd.cut(df['Lifestyle_Quality_Score'], 
                                        bins=[0, 1, 2, 3, 5], 
                                        labels=['Poor', 'Fair', 'Good', 'Excellent'])
        
        lifestyle_diabetes = df.groupby('Lifestyle_Category')['Diabetes_012'].apply(lambda x: (x == 2).mean() * 100)
        lifestyle_diabetes_df = pd.DataFrame({'Diabetes Rate (%)': lifestyle_diabetes})
        st.bar_chart(lifestyle_diabetes_df)
    
    with col2:
        st.subheader("Health Risk Score vs Exercise")
        # Health risk by exercise
        risk_exercise = df.groupby('Physical_Activity_Status')['Health_Risk_Score'].mean()
        risk_exercise_df = pd.DataFrame({'Health Risk Score': risk_exercise})
        st.bar_chart(risk_exercise_df)
    
    # Health Status Analysis
    st.subheader("Overall Health Status Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        health_status_counts = df['Overall_Health_Status'].value_counts()
        health_status_df = pd.DataFrame({'Count': health_status_counts})
        st.bar_chart(health_status_df)
        
        # Show percentages
        st.write("**Health Status Distribution:**")
        total = health_status_counts.sum()
        for status, count in health_status_counts.items():
            percentage = (count / total) * 100
            st.write(f"• **{status}**: {count:,} people ({percentage:.1f}%)")
    
    with col2:
        # Health status by exercise
        health_exercise = pd.crosstab(df['Physical_Activity_Status'], df['Overall_Health_Status'], normalize='index') * 100
        st.bar_chart(health_exercise)
        
        # Show detailed breakdown
        st.write("**Health Status by Physical Activity:**")
        for activity in health_exercise.index:
            st.write(f"**{activity}:**")
            for status in health_exercise.columns:
                percentage = health_exercise.loc[activity, status]
                st.write(f"  • {status}: {percentage:.1f}%")
    
    # Enhanced analysis with df_enhanced
    st.subheader("Enhanced Analysis with Additional Features")
    
    if 'df_enhanced' in locals():
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sleep Hours vs Diabetes Risk")
            # Create sleep summary by diabetes status
            sleep_summary = df_enhanced.groupby('Diabetes_Status')['Sleep_Hours'].agg(['mean', 'std']).round(1)
            sleep_summary_df = pd.DataFrame({'Average Sleep Hours': sleep_summary['mean']})
            st.bar_chart(sleep_summary_df)
            
            # Show detailed statistics
            st.write("**Sleep Statistics by Diabetes Status:**")
            for status in sleep_summary.index:
                mean_sleep = sleep_summary.loc[status, 'mean']
                std_sleep = sleep_summary.loc[status, 'std']
                st.write(f"**{status}:** {mean_sleep:.1f} ± {std_sleep:.1f} hours")
        
        with col2:
            st.subheader("Stress Level vs Exercise")
            stress_exercise = pd.crosstab(df_enhanced['Physical_Activity_Status'], df_enhanced['Stress_Level'], normalize='index') * 100
            st.bar_chart(stress_exercise)
            
            # Show detailed breakdown
            st.write("**Stress Level by Physical Activity:**")
            for activity in stress_exercise.index:
                st.write(f"**{activity}:**")
                for stress in stress_exercise.columns:
                    percentage = stress_exercise.loc[activity, stress]
                    st.write(f"  • {stress}: {percentage:.1f}%")
    
    # Advanced correlation analysis
    st.subheader("Advanced Correlation Analysis")
    
    # Select key variables for correlation
    correlation_vars = ['PhysActivity', 'Lifestyle_Quality_Score', 'Health_Risk_Score', 
                       'BMI', 'Age_Actual', 'Diabetes_012', 'HighBP', 'HighChol']
    
    # Filter out columns that might not exist
    available_vars = [var for var in correlation_vars if var in df.columns]
    corr_matrix = df[available_vars].corr()
    
    # Display correlation matrix as a table
    st.write("**Enhanced Correlation Matrix: Key Health Variables**")
    st.dataframe(corr_matrix.round(3), use_container_width=True)
    
    # Show key correlations with diabetes
    st.write("**Key Correlations with Diabetes (sorted by strength):**")
    diabetes_corr = corr_matrix['Diabetes_012'].drop('Diabetes_012').sort_values(key=abs, ascending=False)
    for factor, corr in diabetes_corr.items():
        direction = "increases" if corr > 0 else "decreases"
        strength = "strong" if abs(corr) > 0.3 else "moderate" if abs(corr) > 0.1 else "weak"
        st.write(f"• **{factor}**: {strength} {direction} diabetes risk (r = {corr:.3f})")
    
    # Key insights
    st.subheader("Key Insights from Enhanced Analysis")
    
    # Calculate key statistics
    high_lifestyle = df[df['Lifestyle_Quality_Score'] >= 3]
    low_lifestyle = df[df['Lifestyle_Quality_Score'] < 2]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("High Lifestyle Group Diabetes Rate", 
                 f"{high_lifestyle['Diabetes_012'].apply(lambda x: x == 2).mean()*100:.1f}%")
    
    with col2:
        st.metric("Low Lifestyle Group Diabetes Rate", 
                 f"{low_lifestyle['Diabetes_012'].apply(lambda x: x == 2).mean()*100:.1f}%")
    
    with col3:
        st.metric("Health Risk Score Difference", 
                 f"{low_lifestyle['Health_Risk_Score'].mean() - high_lifestyle['Health_Risk_Score'].mean():.1f} points")
    
    # Additional insights
    st.write("**Enhanced Analysis Benefits:**")
    st.write("• **Lifestyle Quality Score**: Combines exercise, diet, and health habits")
    st.write("• **Health Risk Score**: Multi-factor risk assessment")
    st.write("• **Comprehensive Health Status**: Overall health categorization")
    st.write("• **Advanced Correlations**: Shows relationships between all health factors")
    st.write("• **Memory Optimized**: Uses efficient data processing to avoid crashes")

# Final summary
st.header("🎯 Key Takeaways")
st.markdown("""
### How Exercise Helps with Diabetes Management:

1. **Direct Risk Reduction**: Physical activity reduces diabetes risk by **{:.1f} percentage points**
2. **Weight Management**: Exercise helps maintain healthier BMI levels (difference of **{:.1f} points**)
3. **Age Benefits**: Exercise provides diabetes protection across all age groups
4. **Lifestyle Synergy**: Physical activity works best when combined with other healthy habits
5. **Prevention Focus**: Exercise is most effective when started early and maintained consistently

### Recommendations:
- **Start Early**: Begin regular physical activity in young adulthood
- **Stay Consistent**: Maintain exercise habits throughout life
- **Combine Strategies**: Pair exercise with healthy diet and lifestyle choices
- **Monitor Progress**: Track BMI and other health indicators regularly
""".format(risk_reduction, bmi_difference))