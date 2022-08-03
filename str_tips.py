from turtle import color
import altair as alt
import streamlit as st
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff
import seaborn as sns
import mpld3
import streamlit.components.v1 as components

path = '/home/gulnaz/ds-phase-0/learning/datasets/tips.csv'
tips = pd.read_csv(path)

def altair_scatter(dataset, x, y):
    plot = (
        alt.Chart(dataset,height=600)
        .mark_point(filled=True, opacity=0.8)
        .encode(x=x, y=y)
    )
    return plot

def mpl_scatter(dataset, x, y):
    fig, ax = plt.subplots()
    dataset.plot.scatter(x=x, y=y, alpha=0.8, ax=ax)
    return fig



st.write("""
# Анализ датасета tips

## Гистограмма общего счета

### matplotlib
 """)
fig,ax = plt.subplots(figsize = (15,10))
#plt.title('Гистограмма общего счета',fontsize = 20,pad =15)
plt.hist(tips['total_bill'],bins=20)
plt.xlabel('total bill')
plt.ylabel('count of total bills in range');
st.pyplot(fig)

st.write("""
### streamlit
 """)

hist_values = np.histogram(tips['total_bill'], bins=20)[0]
st.bar_chart(hist_values)

st.write("""
### plotly
 """)
hist_data = [tips['total_bill']]
group_labels = ['total_bill']
fig = ff.create_distplot(hist_data, group_labels, bin_size=[2]);
st.plotly_chart(fig)

st.write("""
### seaborn
 """)
fig, ax = plt.subplots()
fig = plt.figure(figsize=(20,10))
plt.ticklabel_format(style='plain')
ax = sns.histplot(x = tips['total_bill'])
sns.set(style='darkgrid');
st.pyplot(fig)


st.write("""


## График разброса общего счета/ чаевых

### matplotlib
 """)
mpl_plot = mpl_scatter(tips, "total_bill", "tip")
st.pyplot(mpl_plot)

st.write("""
### matplotlib interactive
 """)
fig = mpl_scatter(tips, "total_bill", "tip")
fig_html = mpld3.fig_to_html(fig)
components.html(fig_html, height=600);

st.write("""
### altair
 """)
alt_plot = altair_scatter(tips, "total_bill", "tip")
st.altair_chart(alt_plot.interactive(),use_container_width=True);

st.write("""
### seaborn jointplot
 """)
fig, ax = plt.subplots()
fig = sns.jointplot(x=tips["total_bill"], y=tips["tip"], kind='hex')
st.pyplot(fig);

st.write("""


## График разброса общего счета/размера чаевых

### matplotlib
 """)
fig,ax = plt.subplots(figsize = (15,10))
sc = plt.scatter(tips['total_bill'],tips['tip'],c = tips['size'],s=100,alpha =0.6)
plt.xlabel('total bill')
plt.ylabel('tip')
plt.title('График разброса общего счета/размера чаевых',fontsize = 20,pad =15)
plt.legend(*sc.legend_elements("colors", num=5,size=10),title="size")
st.pyplot(fig);


st.write("""
### altair
 """)
domain_ = [i for i in range (tips['size'].min(),tips['size'].max()+1)]
range_ = ['yellow','orange','red','green', 'blue', 'black']

alt_plot2 = alt.Chart(tips,height=600).mark_point(size = 60).encode(
    x='total_bill',
    y='tip',
    color=alt.Color('size', scale=alt.Scale(domain=domain_, range=range_)),
).configure_mark(
    filled=True,opacity=0.5
)
st.altair_chart(alt_plot2.interactive(),use_container_width=True)


st.write("""


## Связь между днем недели и размером счета

### seaborn catplot
 """)
fig,ax = plt.subplots(figsize = (15,10))
fig = sns.catplot(x='day',
            y='total_bill',
            data=tips);
st.pyplot(fig)


st.write("""
### seaborn boxplot
 """)
fig,ax = plt.subplots(figsize = (15,10))
fig = sns.catplot(x='day',
            y='total_bill',
            kind='box',
            data=tips);
st.pyplot(fig)



st.write("""
### matplotlib histograms
 """)
fig,ax = plt.subplots(figsize = (15,10))

kwargs = dict(histtype='stepfilled', 
              alpha=0.3, 
              #density=True, 
              bins=20)

for day in tips['day'].unique():
    plt.hist(tips[tips['day']==day]['total_bill'], **kwargs, label=day)
plt.legend(title='Day')
plt.title('Гистограммы total bill для каждого представленного дня недели',fontsize = 20, pad=15)
plt.xlabel('total bill')
plt.ylabel('count of total bills in range');
st.pyplot(fig)


st.write("""


## График разброса дня недели/чаевых

### matplotlib
 """)
fig,ax = plt.subplots(figsize = (15,10))
plt.xlabel('tip')
plt.ylabel('day')
sc = ax.scatter(tips['tip'], tips['day'], c=tips.sex.astype('category').cat.codes,s=200,alpha =0.8)
plt.legend(handles=sc.legend_elements()[0], 
           labels=['Female','Male'],
           title="sex")
plt.title('График разброса дня недели/чаевых',fontsize = 20,pad=15);
st.pyplot(fig)


st.write("""
### altair
 """)
alt_plot = alt.Chart(tips,height=600).mark_point(filled=True, opacity=0.4,size=80).encode(x='tip', y='day',color =alt.Color('sex'));
st.altair_chart(alt_plot.interactive(),use_container_width=True)


st.write("""


## Box plot c суммой всех счетов за каждый день, с разбитием по time (Dinner/Lunch)

### seaborn
 """)
fig,ax = plt.subplots(figsize = (15,10))
fig = sns.catplot(x ="day", y = "total_bill", hue = "time", data = tips, kind="box");
st.pyplot(fig)



st.write("""


## Гистограммы чаевых на обед и ланч

### seaborn
 """)
fig = sns.FacetGrid (tips, col = "time",height=8) 
fig.map(plt.hist, "tip");
st.pyplot(fig)


st.write("""


## Графики разброса размера счета и чаевых для мужчин и женщин, с дополнительным разбиением по курящим/некурящим.

### seaborn
 """)
fig,ax = plt.subplots(figsize = (15,10))
fig = sns.relplot(x='total_bill',
            y='tip',
            col='sex',
            hue = 'smoker',
            data=tips);
st.pyplot(fig)