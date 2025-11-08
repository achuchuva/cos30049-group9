import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import { Doughnut } from 'react-chartjs-2';
import axios from 'axios'

import { useEffect, useState } from 'react';

const API_BASE = 'http://localhost:8000';

ChartJS.register(ArcElement, Tooltip, Legend);

// export const data = {
//     labels: ['Red', 'Blue', 'Yellow', 'Green', 'Purple', 'Orange'],
//     datasets: [
//         {
//             label: '# of Votes',
//             data: [12, 19, 3, 5, 2, 3],
//             backgroundColor: [
//                 'rgba(255, 99, 132, 0.2)',
//                 'rgba(54, 162, 235, 0.2)',
//                 'rgba(255, 206, 86, 0.2)',
//                 'rgba(75, 192, 192, 0.2)',
//                 'rgba(153, 102, 255, 0.2)',
//                 'rgba(255, 159, 64, 0.2)',
//             ],
//             borderColor: [
//                 'rgba(255, 99, 132, 1)',
//                 'rgba(54, 162, 235, 1)',
//                 'rgba(255, 206, 86, 1)',
//                 'rgba(75, 192, 192, 1)',
//                 'rgba(153, 102, 255, 1)',
//                 'rgba(255, 159, 64, 1)',
//             ],
//             borderWidth: 1,
//         },
//     ],
// };

function ChartjsDoughnut() {
  const [stats, setStats] = useState([null]);
  const [error, setError] = useState(null);

  useEffect(()=>{
        axios.get(`${API_BASE}/api/v1/stats`)
            .then((response)=>{
                if(response.status===200){   
                    console.log(response.data)
                    setStats(response.data)
                }
                else{
                    setError("ERROR: "+response.status+" , "+response.statusText)
                }
            })
            .catch((err) => console.log.err);
    },[]);
  
  const data = {
      labels: ['Total predictions', 'Spam', 'Ham'],
      datasets: [
          {
              label: 'Statistics',
              data: [stats.total_predictions, stats.spam_count, stats.ham_count],
              backgroundColor: [
                  'rgba(168, 168, 168, 0.82)',
                  'rgba(231, 45, 12, 0.82)',
                  'rgba(123, 255, 0, 0.6)',
              ],
              borderColor: [
                  'rgba(63, 63, 63, 1)',
                  'rgba(235, 54, 54, 1)',
                  'rgba(125, 255, 86, 1)',
              ],
              borderWidth: 1,
          },
      ],
  };

  // return <Doughnut data={data} />;
  return (
    <section>
      <Doughnut data={data} />
    </section>
  );
}

export default ChartjsDoughnut;