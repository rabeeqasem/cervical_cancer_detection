import React from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';
import { Bar } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const options = {
  responsive: true,
  plugins: {
    legend: {
      display: false
    },
    title: {
      display: true,
      text: 'Prediction Probabilities',
    },
  },
  maintainAspectRatio: false,
  scale: {
    y: {
      min: 0,
      max: 100
    }
  }
};


const Chart = ({probabilities}) => {
  const data = {
    labels: ['NILM', 'ASC-US', 'ASC-H', 'LSIL', 'HSIL', 'SCC'],
    datasets: [
      { 
        label: 'Probability',
        data: probabilities ? probabilities[0] : [0, 0, 0, 0, 0, 0],
        backgroundColor: [
          'rgba(255, 99, 132, 0.5)',
          'rgba(54, 162, 235, 0.5)',
          'rgba(255, 206, 86, 0.5)',
          'rgba(153, 102, 255, 0.5)',
          'rgba(255, 159, 64, 0.5)',
          'rgba(75, 192, 192, 0.5)'
        ]
      },
    ],
  }

  return (
    <div className="w-80 h-72 p-2 mx-auto mt-10 lg:mt-0 rounded-md shadow-xl">
      <Bar options={options} data={data} />
    </div>
  );
}

export default Chart;