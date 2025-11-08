import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import axios from 'axios';

const API_BASE = 'http://localhost:8000';

export default function D3Implementation() {
    const svgRef = useRef();
    const [stats, setStats] = useState(null);
    const [error, setError] = useState(null);

    useEffect(() => {
        axios.get(`${API_BASE}/api/v1/stats`)
            .then((response) => {
                if (response.status === 200) {
                    console.log(response.data);
                    setStats(response.data);
                }
                else {
                    setError("ERROR: " + response.status + " , " + response.statusText);
                }
            })
            .catch((err) => {
                console.error(err);
                setError("Failed to load stats");
            });
    }, []);

    useEffect(() => {
        if (!stats || !stats.feature_distribution) return;

        // Clear previous SVG content
        d3.select(svgRef.current).selectAll("*").remove();

        const data = stats.feature_distribution;
        const width = 900;
        const height = 600;
        const margin = { top: 60, right: 150, bottom: 80, left: 80 };

        // Create SVG
        const svg = d3.select(svgRef.current)
            .attr('width', width)
            .attr('height', height);

        // Create scales with better domains
        const xScale = d3.scaleLinear()
            .domain([0, d3.max(data, d => d.word_count) * 1.1])
            .range([margin.left, width - margin.right]);

        const yScale = d3.scaleLinear()
            .domain([0, d3.max(data, d => d.suspicious_word_count) * 1.1])
            .range([height - margin.bottom, margin.top]);

        // Size scale based on character count
        const sizeScale = d3.scaleSqrt()
            .domain([0, d3.max(data, d => d.char_count)])
            .range([4, 20]);

        // Create grid
        svg.append('g')
            .attr('class', 'grid')
            .attr('transform', `translate(0,${height - margin.bottom})`)
            .call(d3.axisBottom(xScale).tickSize(-height + margin.top + margin.bottom).tickFormat(''))
            .selectAll('line')
            .style('stroke', '#e5e5e5')
            .style('stroke-opacity', 0.7);

        svg.append('g')
            .attr('class', 'grid')
            .attr('transform', `translate(${margin.left},0)`)
            .call(d3.axisLeft(yScale).tickSize(-width + margin.left + margin.right).tickFormat(''))
            .selectAll('line')
            .style('stroke', '#e5e5e5')
            .style('stroke-opacity', 0.7);

        // Create axes
        const xAxis = d3.axisBottom(xScale).ticks(10);
        const yAxis = d3.axisLeft(yScale).ticks(10);

        svg.append('g')
            .attr('transform', `translate(0,${height - margin.bottom})`)
            .call(xAxis)
            .selectAll('text')
            .style('font-size', '12px');

        svg.append('g')
            .attr('transform', `translate(${margin.left},0)`)
            .call(yAxis)
            .selectAll('text')
            .style('font-size', '12px');

        // Add axis labels
        svg.append('text')
            .attr('x', width / 2)
            .attr('y', height - 20)
            .attr('text-anchor', 'middle')
            .style('font-size', '14px')
            .style('font-weight', 'bold')
            .text('Word Count');

        svg.append('text')
            .attr('transform', 'rotate(-90)')
            .attr('x', -height / 2)
            .attr('y', 20)
            .attr('text-anchor', 'middle')
            .style('font-size', '14px')
            .style('font-weight', 'bold')
            .text('Suspicious Word Count');

        // Add title
        svg.append('text')
            .attr('x', width / 2)
            .attr('y', 30)
            .attr('text-anchor', 'middle')
            .style('font-size', '18px')
            .style('font-weight', 'bold')
            .text('Message Feature Analysis');

        svg.append('text')
            .attr('x', width / 2)
            .attr('y', 50)
            .attr('text-anchor', 'middle')
            .style('font-size', '12px')
            .style('fill', '#666')
            .text('Bubble size represents character count');

        // Create tooltip
        const tooltip = d3.select('body').append('div')
            .style('position', 'absolute')
            .style('padding', '10px')
            .style('background', 'rgba(0, 0, 0, 0.85)')
            .style('color', '#fff')
            .style('border-radius', '6px')
            .style('font-size', '13px')
            .style('pointer-events', 'none')
            .style('opacity', 0)
            .style('box-shadow', '0 4px 8px rgba(0,0,0,0.2)');

        // Add circles with animation
        svg.selectAll('circle')
            .data(data)
            .enter()
            .append('circle')
            .attr('cx', d => xScale(d.word_count))
            .attr('cy', d => yScale(d.suspicious_word_count))
            .attr('r', 0)
            .attr('fill', d => d.is_spam ? '#e72d0c' : '#7dff56')
            .attr('stroke', d => d.is_spam ? '#b32209' : '#4fc22e')
            .attr('stroke-width', 2)
            .attr('opacity', 0.6)
            .on('mouseover', function(event, d) {
                d3.select(this)
                    .transition()
                    .duration(200)
                    .attr('opacity', 1)
                    .attr('stroke-width', 3);
                
                tooltip.transition().duration(200).style('opacity', 1);
                tooltip.html(`
                    <div style="font-weight: bold; margin-bottom: 6px; color: ${d.is_spam ? '#ff6b6b' : '#7dff56'};">
                        ${d.is_spam ? '⚠️ SPAM' : '✅ HAM'}
                    </div>
                    <div style="line-height: 1.6;">
                        <strong>Word Count:</strong> ${d.word_count}<br/>
                        <strong>Suspicious Words:</strong> ${d.suspicious_word_count}<br/>
                        <strong>Characters:</strong> ${d.char_count}<br/>
                        <strong>URLs:</strong> ${d.url_count}
                    </div>
                `)
                    .style('left', (event.pageX + 15) + 'px')
                    .style('top', (event.pageY - 28) + 'px');
            })
            .on('mouseout', function() {
                d3.select(this)
                    .transition()
                    .duration(200)
                    .attr('opacity', 0.6)
                    .attr('stroke-width', 2);
                
                tooltip.transition().duration(300).style('opacity', 0);
            })
            .transition()
            .duration(800)
            .delay((d, i) => i * 5)
            .attr('r', d => sizeScale(d.char_count));

        // Add legend
        const legend = svg.append('g')
            .attr('transform', `translate(${width - 130}, ${margin.top + 20})`);

        // Spam legend
        legend.append('circle')
            .attr('cx', 0)
            .attr('cy', 0)
            .attr('r', 8)
            .attr('fill', '#e72d0c')
            .attr('stroke', '#b32209')
            .attr('stroke-width', 2)
            .attr('opacity', 0.6);

        legend.append('text')
            .attr('x', 20)
            .attr('y', 5)
            .text('Spam')
            .style('font-size', '13px')
            .style('font-weight', 'bold');

        // Ham legend
        legend.append('circle')
            .attr('cx', 0)
            .attr('cy', 30)
            .attr('r', 8)
            .attr('fill', '#7dff56')
            .attr('stroke', '#4fc22e')
            .attr('stroke-width', 2)
            .attr('opacity', 0.6);

        legend.append('text')
            .attr('x', 20)
            .attr('y', 35)
            .text('Ham')
            .style('font-size', '13px')
            .style('font-weight', 'bold');

        // Size legend
        const sizeLegend = svg.append('g')
            .attr('transform', `translate(${width - 130}, ${margin.top + 80})`);

        sizeLegend.append('text')
            .attr('x', 0)
            .attr('y', 0)
            .text('Character Count:')
            .style('font-size', '11px')
            .style('font-weight', 'bold')
            .style('fill', '#666');

        [25, 50, 75].forEach((percentile, i) => {
            const maxChars = d3.max(data, d => d.char_count);
            const charCount = Math.round(maxChars * (percentile / 100));
            const radius = sizeScale(charCount);
            
            sizeLegend.append('circle')
                .attr('cx', 0)
                .attr('cy', 25 + i * 25)
                .attr('r', radius)
                .attr('fill', 'none')
                .attr('stroke', '#999')
                .attr('stroke-width', 1);
            
            sizeLegend.append('text')
                .attr('x', 25)
                .attr('y', 25 + i * 25 + 4)
                .text(`${charCount} chars`)
                .style('font-size', '10px')
                .style('fill', '#666');
        });

        // Cleanup tooltip on unmount
        return () => {
            tooltip.remove();
        };
    }, [stats]);

    const exportAsPNG = () => {
        const svgElement = svgRef.current;
        const serializer = new XMLSerializer();
        const svgString = serializer.serializeToString(svgElement);
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const img = new Image();
        
        canvas.width = 900;
        canvas.height = 600;
        
        img.onload = () => {
            ctx.fillStyle = 'white';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0);
            const pngUrl = canvas.toDataURL('image/png');
            const link = document.createElement('a');
            link.download = 'feature-analysis.png';
            link.href = pngUrl;
            link.click();
        };
        
        img.src = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svgString)));
    };

    if (error) {
        return <div className="error">{error}</div>;
    }

    if (!stats) {
        return <div>Loading...</div>;
    }

    return (
        <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <div>
                    <h2 style={{ margin: '0 0 8px 0' }}>Feature Correlation Analysis</h2>
                    <p style={{ fontSize: '0.85rem', color: '#666', margin: 0 }}>
                        Interactive bubble chart showing relationship between word count, suspicious words, and message length
                    </p>
                </div>
                <button 
                    className="btn" 
                    onClick={exportAsPNG}
                    style={{ fontSize: '0.85rem' }}
                >
                    Export Chart
                </button>
            </div>
            <div style={{ display: 'flex', justifyContent: 'center', marginTop: '1.5rem' }}>
                <svg ref={svgRef}></svg>
            </div>
        </div>
    );
}
