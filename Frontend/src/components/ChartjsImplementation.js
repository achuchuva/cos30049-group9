//chart.js stuff
import {CategoryScale, Chart, LinearScale, LineController, LineElement, PointElement} from 'chart.js';
import {Canvas} from 'skia-canvas';

export default function Chartjs() {
    
    Chart.register([
    CategoryScale,
    LineController,
    LineElement,
    LinearScale,
    PointElement
    ])

    const canvas = new Canvas(400, 300)
    const chart = new Chart(
    canvas,
    {
        type: 'line',
        data: {
            labels: ['Red', 'Blue', 'Yellow', 'Green', 'Purple', 'Orange'],
            datasets: [{
            label: '# of Votes',
            data: [12, 19, 3, 5, 2, 3],
            borderColor: 'red'
            }]
        }
    }
    );
    chart.destroy();
}