/**
 * Visualization module for the Solana Token Analysis Agent Swarm.
 * Integrates D3.js, ECharts, and TradingView Lightweight Charts for advanced visualizations.
 */

// Import visualization libraries
// These need to be installed via npm:
// npm install d3 echarts lightweight-charts

/**
 * TokenRelationshipGraph - Uses D3.js to visualize token and wallet relationships
 * Helps identify related tokens, risky wallet clusters, and liquidity patterns
 */
class TokenRelationshipGraph {
  constructor(containerId, width = 900, height = 600) {
    this.containerId = containerId;
    this.width = width;
    this.height = height;
    this.svg = null;
    this.simulation = null;
    this.nodes = [];
    this.links = [];
  }

  initialize() {
    if (typeof d3 === 'undefined') {
      console.error('D3.js is required for TokenRelationshipGraph');
      return;
    }

    const container = document.getElementById(this.containerId);
    if (!container) {
      console.error(`Container with ID ${this.containerId} not found`);
      return;
    }

    // Clear any existing content
    container.innerHTML = '';

    // Create SVG container
    this.svg = d3.select(`#${this.containerId}`)
      .append('svg')
      .attr('width', this.width)
      .attr('height', this.height);
      
    // Add zoom functionality
    const zoom = d3.zoom()
      .scaleExtent([0.1, 10])
      .on('zoom', (event) => {
        this.svg.selectAll('g.graph-container').attr('transform', event.transform);
      });
      
    this.svg.call(zoom);
    
    // Create container for graph elements that will be zoomed
    this.svg.append('g')
      .attr('class', 'graph-container');
      
    // Initialize force simulation
    this.simulation = d3.forceSimulation()
      .force('link', d3.forceLink().id(d => d.id).distance(100))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(this.width / 2, this.height / 2))
      .force('collision', d3.forceCollide().radius(40));
    
    // Add legend
    this.addLegend();
  }
  
  addLegend() {
    const legendData = [
      { type: 'token', label: 'Token' },
      { type: 'wallet', label: 'Wallet' },
      { type: 'contract', label: 'Contract' },
      { type: 'pool', label: 'Liquidity Pool' },
      { type: 'scam', label: 'Suspicious Entity' }
    ];
    
    const legend = this.svg.append('g')
      .attr('class', 'legend')
      .attr('transform', 'translate(20, 20)');
      
    const legendItems = legend.selectAll('.legend-item')
      .data(legendData)
      .enter()
      .append('g')
      .attr('class', 'legend-item')
      .attr('transform', (d, i) => `translate(0, ${i * 25})`);
      
    legendItems.append('circle')
      .attr('r', 7)
      .attr('fill', d => this.getColorByType(d.type));
      
    legendItems.append('text')
      .attr('x', 15)
      .attr('y', 5)
      .text(d => d.label)
      .attr('font-size', '12px');
  }
  
  render(nodes, links) {
    this.nodes = nodes;
    this.links = links;
    
    // Get graph container
    const container = this.svg.select('g.graph-container');
    
    // Clear previous visualization elements
    container.selectAll('*').remove();
    
    // Create links
    const link = container.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(links)
      .enter()
      .append('line')
      .attr('stroke-width', d => Math.sqrt(d.value || 1))
      .attr('stroke', '#999')
      .attr('stroke-opacity', 0.6);
      
    // Create nodes
    const node = container.append('g')
      .attr('class', 'nodes')
      .selectAll('.node')
      .data(nodes)
      .enter()
      .append('g')
      .attr('class', 'node')
      .call(d3.drag()
        .on('start', this.dragstarted.bind(this))
        .on('drag', this.dragged.bind(this))
        .on('end', this.dragended.bind(this)));
        
    // Add circles to nodes
    node.append('circle')
      .attr('r', d => 10 + Math.sqrt(d.value || 5))
      .attr('fill', d => this.getColorByType(d.type));
      
    // Add labels to nodes
    node.append('text')
      .attr('dy', 4)
      .attr('text-anchor', 'middle')
      .text(d => d.symbol || d.label || d.id.substring(0, 6))
      .attr('font-size', '10px')
      .attr('fill', '#fff');
      
    // Add tooltips
    node.append('title')
      .text(d => {
        const lines = [
          `${d.name || d.id}`,
          `Type: ${d.type}`,
          `Value: ${d.value || 'N/A'}`
        ];
        
        if (d.type === 'token' && d.holders) {
          lines.push(`Holders: ${d.holders}`);
        }
        
        if (d.type === 'wallet' && d.balance) {
          lines.push(`Balance: ${d.balance}`);
        }
        
        return lines.join('\n');
      });
      
    // Update simulation
    this.simulation
      .nodes(nodes)
      .on('tick', () => {
        link
          .attr('x1', d => d.source.x)
          .attr('y1', d => d.source.y)
          .attr('x2', d => d.target.x)
          .attr('y2', d => d.target.y);
          
        node
          .attr('transform', d => `translate(${d.x},${d.y})`);
      });
      
    this.simulation.force('link').links(links);
    this.simulation.alpha(1).restart();
  }
  
  dragstarted(event, d) {
    if (!event.active) this.simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
  }
  
  dragged(event, d) {
    d.fx = event.x;
    d.fy = event.y;
  }
  
  dragended(event, d) {
    if (!event.active) this.simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
  }
  
  getColorByType(type) {
    const colorMap = {
      'token': '#4e79a7',
      'wallet': '#f28e2c',
      'contract': '#59a14f',
      'pool': '#e15759',
      'exchange': '#76b7b2',
      'scam': '#e31a1c'
    };
    
    return colorMap[type] || '#8c564b';
  }
  
  addNode(node) {
    // Check if node already exists
    const exists = this.nodes.some(n => n.id === node.id);
    if (!exists) {
      this.nodes.push(node);
      this.render(this.nodes, this.links);
    }
  }
  
  addLink(link) {
    // Check if link already exists
    const exists = this.links.some(l => 
      (l.source.id === link.source && l.target.id === link.target) || 
      (l.source.id === link.target && l.target.id === link.source)
    );
    if (!exists) {
      this.links.push(link);
      this.render(this.nodes, this.links);
    }
  }
  
  highlightNode(nodeId) {
    this.svg.selectAll('.node circle')
      .attr('stroke', d => d.id === nodeId ? '#ff0' : null)
      .attr('stroke-width', d => d.id === nodeId ? 3 : 0);
  }
  
  reset() {
    this.nodes = [];
    this.links = [];
    this.render(this.nodes, this.links);
  }
}

/**
 * TokenometricsDashboard - Uses ECharts to create comprehensive token analytics dashboard
 * Displays price, volume, liquidity, and risk metrics in an interactive dashboard
 */
class TokenometricsDashboard {
  constructor() {
    this.charts = {};
    this.resizeHandler = null;
  }
  
  initialize(containerId) {
    if (typeof echarts === 'undefined') {
      console.error('ECharts is required for TokenometricsDashboard');
      return;
    }
    
    const container = document.getElementById(containerId);
    if (!container) {
      console.error(`Container with ID ${containerId} not found`);
      return;
    }
    
    // Clear any existing content
    container.innerHTML = '';
    
    // Create grid layout
    const layout = [
      { id: 'price-chart', title: 'Price History', width: '100%', height: '300px' },
      { id: 'volume-chart', title: 'Volume', width: '50%', height: '200px' },
      { id: 'liquidity-chart', title: 'Liquidity', width: '50%', height: '200px' },
      { id: 'holder-distribution', title: 'Holder Distribution', width: '50%', height: '250px' },
      { id: 'risk-radar', title: 'Risk Assessment', width: '50%', height: '250px' }
    ];
    
    // Create chart containers
    layout.forEach(item => {
      const chartContainer = document.createElement('div');
      chartContainer.id = item.id;
      chartContainer.style.width = item.width;
      chartContainer.style.height = item.height;
      chartContainer.style.float = 'left';
      chartContainer.style.padding = '10px';
      chartContainer.style.boxSizing = 'border-box';
      
      const titleEl = document.createElement('h3');
      titleEl.textContent = item.title;
      titleEl.style.margin = '0 0 10px 0';
      titleEl.style.fontSize = '14px';
      
      const chartEl = document.createElement('div');
      chartEl.style.width = '100%';
      chartEl.style.height = 'calc(100% - 24px)';
      chartEl.className = 'chart-container';
      
      chartContainer.appendChild(titleEl);
      chartContainer.appendChild(chartEl);
      container.appendChild(chartContainer);
      
      // Initialize chart
      this.charts[item.id] = echarts.init(chartEl);
    });
    
    // Add window resize handler
    this.resizeHandler = () => {
      for (const chart of Object.values(this.charts)) {
        chart.resize();
      }
    };
    
    window.addEventListener('resize', this.resizeHandler);
  }
  
  renderPriceChart(data) {
    if (!this.charts['price-chart']) return;
    
    const option = {
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'cross'
        }
      },
      legend: {
        data: ['Price', 'MA5', 'MA20']
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
      },
      xAxis: {
        type: 'time',
        boundaryGap: false
      },
      yAxis: {
        type: 'value',
        scale: true
      },
      dataZoom: [
        { type: 'inside', start: 0, end: 100 },
        { type: 'slider', start: 0, end: 100 }
      ],
      series: [
        {
          name: 'Price',
          type: 'line',
          smooth: true,
          symbol: 'none',
          sampling: 'lttb',
          areaStyle: {
            opacity: 0.3
          },
          data: data.map(item => [item.timestamp, item.price])
        },
        {
          name: 'MA5',
          type: 'line',
          smooth: true,
          symbol: 'none',
          data: this.calculateMA(5, data)
        },
        {
          name: 'MA20',
          type: 'line',
          smooth: true,
          symbol: 'none',
          data: this.calculateMA(20, data)
        }
      ]
    };
    
    this.charts['price-chart'].setOption(option);
  }
  
  calculateMA(dayCount, data) {
    const result = [];
    for (let i = 0; i < data.length; i++) {
      if (i < dayCount - 1) {
        result.push([data[i].timestamp, '-']);
        continue;
      }
      let sum = 0;
      for (let j = 0; j < dayCount; j++) {
        sum += data[i - j].price;
      }
      result.push([data[i].timestamp, (sum / dayCount).toFixed(4)]);
    }
    return result;
  }
  
  renderVolumeChart(data) {
    if (!this.charts['volume-chart']) return;
    
    const option = {
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'shadow'
        }
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
      },
      xAxis: {
        type: 'time',
        boundaryGap: true
      },
      yAxis: {
        type: 'value'
      },
      series: [
        {
          name: 'Volume',
          type: 'bar',
          data: data.map(item => [item.timestamp, item.volume]),
          itemStyle: {
            color: function(params) {
              // Color bars based on price direction
              return params.value[1] > 0 
                ? (data[params.dataIndex].direction === 'up' ? '#c23531' : '#3a9a45') 
                : '#2f4554';
            }
          }
        }
      ]
    };
    
    this.charts['volume-chart'].setOption(option);
  }
  
  renderLiquidityChart(data) {
    if (!this.charts['liquidity-chart']) return;
    
    const option = {
      tooltip: {
        trigger: 'axis'
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
      },
      xAxis: {
        type: 'time',
        boundaryGap: false
      },
      yAxis: {
        type: 'value'
      },
      series: [
        {
          name: 'Liquidity',
          type: 'line',
          smooth: true,
          symbolSize: 5,
          sampling: 'lttb',
          itemStyle: {
            color: '#80FFA5'
          },
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              {
                offset: 0,
                color: 'rgba(128, 255, 165, 0.5)'
              },
              {
                offset: 1,
                color: 'rgba(1, 191, 236, 0.1)'
              }
            ])
          },
          data: data.map(item => [item.timestamp, item.liquidity])
        }
      ]
    };
    
    this.charts['liquidity-chart'].setOption(option);
  }
  
  renderHolderDistribution(data) {
    if (!this.charts['holder-distribution']) return;
    
    const option = {
      tooltip: {
        trigger: 'item',
        formatter: '{a} <br/>{b}: {c} ({d}%)'
      },
      legend: {
        orient: 'vertical',
        left: 10,
        data: data.map(item => item.category)
      },
      series: [
        {
          name: 'Holder Distribution',
          type: 'pie',
          radius: ['50%', '70%'],
          avoidLabelOverlap: false,
          itemStyle: {
            borderRadius: 10,
            borderColor: '#fff',
            borderWidth: 2
          },
          label: {
            show: false,
            position: 'center'
          },
          emphasis: {
            label: {
              show: true,
              fontSize: '14',
              fontWeight: 'bold'
            }
          },
          labelLine: {
            show: false
          },
          data: data.map(item => ({
            value: item.percentage,
            name: item.category
          }))
        }
      ]
    };
    
    this.charts['holder-distribution'].setOption(option);
  }
  
  renderRiskRadar(data) {
    if (!this.charts['risk-radar']) return;
    
    const option = {
      tooltip: {},
      radar: {
        indicator: data.factors.map(factor => ({
          name: factor.name,
          max: 1
        }))
      },
      series: [
        {
          name: 'Risk Factors',
          type: 'radar',
          data: [
            {
              value: data.factors.map(factor => factor.score),
              name: 'Risk Assessment',
              areaStyle: {
                color: 'rgba(255, 99, 71, 0.6)'
              }
            }
          ]
        }
      ]
    };
    
    this.charts['risk-radar'].setOption(option);
  }
  
  updateCharts(tokenData) {
    // Update all charts with new data
    if (tokenData.priceHistory) {
      this.renderPriceChart(tokenData.priceHistory);
    }
    
    if (tokenData.volumeHistory) {
      this.renderVolumeChart(tokenData.volumeHistory);
    }
    
    if (tokenData.liquidityHistory) {
      this.renderLiquidityChart(tokenData.liquidityHistory);
    }
    
    if (tokenData.holderDistribution) {
      this.renderHolderDistribution(tokenData.holderDistribution);
    }
    
    if (tokenData.riskAssessment) {
      this.renderRiskRadar(tokenData.riskAssessment);
    }
  }
  
  dispose() {
    // Clean up resources
    if (this.resizeHandler) {
      window.removeEventListener('resize', this.resizeHandler);
    }
    
    for (const chart of Object.values(this.charts)) {
      chart.dispose();
    }
    
    this.charts = {};
  }
}

/**
 * TradingViewChartManager - Integrates TradingView Lightweight Charts for financial charting
 * Provides professional-grade chart visualization with technical analysis capabilities
 */
class TradingViewChartManager {
  constructor(containerId, options = {}) {
    this.containerId = containerId;
    this.options = {
      width: 800,
      height: 400,
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
      },
      ...options
    };
    
    this.chart = null;
    this.series = {};
    this.markers = [];
    this.resizeHandler = null;
  }
  
  initialize() {
    if (typeof LightweightCharts === 'undefined') {
      console.error('TradingView Lightweight Charts is required for TradingViewChartManager');
      return;
    }
    
    const container = document.getElementById(this.containerId);
    if (!container) {
      console.error(`Container with ID ${this.containerId} not found`);
      return;
    }
    
    // Clear any existing content
    container.innerHTML = '';
    
    // Measure container size
    const { width, height } = container.getBoundingClientRect();
    
    // Create chart
    this.chart = LightweightCharts.createChart(container, {
      ...this.options,
      width: width || this.options.width,
      height: height || this.options.height
    });
    
    // Add legend container
    const legendContainer = document.createElement('div');
    legendContainer.className = 'tv-chart-legend';
    legendContainer.style.position = 'absolute';
    legendContainer.style.left = '12px';
    legendContainer.style.top = '12px';
    legendContainer.style.zIndex = '2';
    legendContainer.style.fontSize = '12px';
    legendContainer.style.padding = '8px';
    legendContainer.style.background = 'rgba(255, 255, 255, 0.7)';
    legendContainer.style.borderRadius = '4px';
    container.style.position = 'relative';
    container.appendChild(legendContainer);
    this.legendContainer = legendContainer;
    
    // Add window resize handler
    this.resizeHandler = () => {
      const { width, height } = container.getBoundingClientRect();
      this.chart.applyOptions({ width, height });
    };
    
    window.addEventListener('resize', this.resizeHandler);
    
    // Add toolbar
    this.createToolbar(container);
  }
  
  createToolbar(container) {
    const toolbar = document.createElement('div');
    toolbar.className = 'tv-chart-toolbar';
    toolbar.style.position = 'absolute';
    toolbar.style.right = '12px';
    toolbar.style.top = '12px';
    toolbar.style.zIndex = '2';
    toolbar.style.display = 'flex';
    
    // Patterns button
    const patternsButton = document.createElement('button');
    patternsButton.innerHTML = '📈 Patterns';
    patternsButton.style.marginRight = '5px';
    patternsButton.onclick = () => this.togglePatterns();
    
    // Reset zoom button
    const resetButton = document.createElement('button');
    resetButton.innerHTML = '🔍 Reset';
    resetButton.onclick = () => this.fitContent();
    
    toolbar.appendChild(patternsButton);
    toolbar.appendChild(resetButton);
    container.appendChild(toolbar);
  }
  
  togglePatterns() {
    // Toggle pattern visibility
    if (this.patterns && this.patterns.visible) {
      // Hide patterns
      this.clearMarkers();
      this.patterns.visible = false;
    } else {
      // Show patterns - typically you would call your pattern detection here
      this.detectPatterns();
      this.patterns = { visible: true };
    }
  }
  
  detectPatterns() {
    // This is a placeholder for pattern detection
    // In a real implementation, you would use your ML-based pattern detector
    
    // For demonstration, we'll add some sample patterns
    const randomPatterns = [
      { time: this.getRandomTime(), position: 'belowBar', color: '#2196F3', shape: 'arrowUp', text: 'Buy Signal' },
      { time: this.getRandomTime(), position: 'aboveBar', color: '#FF5252', shape: 'arrowDown', text: 'Sell Signal' },
      { time: this.getRandomTime(), position: 'inBar', color: '#4CAF50', shape: 'circle', text: 'Support Level' }
    ];
    
    this.addMarkers('candlestick', randomPatterns);
  }
  
  getRandomTime() {
    // Helper to get a random time from our chart
    if (!this.series.candlestick || !this.candleData) {
      return Date.now() / 1000;
    }
    
    const randomIndex = Math.floor(Math.random() * this.candleData.length);
    return this.candleData[randomIndex].time;
  }
  
  addCandlestickSeries(data, options = {}) {
    if (!this.chart) return null;
    
    // Store data for later use
    this.candleData = data;
    
    // Create candlestick series
    const series = this.chart.addCandlestickSeries({
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderVisible: false,
      wickUpColor: '#26a69a',
      wickDownColor: '#ef5350',
      ...options
    });
    
    series.setData(data);
    this.series.candlestick = series;
    
    // Update legend with latest price
    this.updateLegend(data[data.length - 1]);
    
    return series;
  }
  
  updateLegend(priceData) {
    if (!this.legendContainer) return;
    
    const { open, high, low, close } = priceData;
    
    this.legendContainer.innerHTML = `
      <div>O: <span style="color: ${close >= open ? '#26a69a' : '#ef5350'}">${open.toFixed(4)}</span></div>
      <div>H: <span style="color: #26a69a">${high.toFixed(4)}</span></div>
      <div>L: <span style="color: #ef5350">${low.toFixed(4)}</span></div>
      <div>C: <span style="color: ${close >= open ? '#26a69a' : '#ef5350'}">${close.toFixed(4)}</span></div>
    `;
  }
  
  addVolumeSeries(data, options = {}) {
    if (!this.chart) return null;
    
    // Create volume series
    const series = this.chart.addHistogramSeries({
      color: '#26a69a',
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: 'volume',
      ...options
    });
    
    // Configure price scale for volume
    this.chart.applyOptions({
      priceScale: {
        volume: {
          scaleMargins: {
            top: 0.8,
            bottom: 0,
          },
          visible: true,
        },
      },
    });
    
    // Process volume data to include colors
    const coloredData = data.map((item, index) => {
      const isGreen = index > 0 ? item.value > data[index - 1].value : true;
      return {
        ...item,
        color: isGreen ? '#26a69a' : '#ef5350'
      };
    });
    
    series.setData(coloredData);
    
    this.series.volume = series;
    return series;
  }
  
  addLineSeries(data, seriesName, options = {}) {
    if (!this.chart) return null;
    
    // Create line series
    const series = this.chart.addLineSeries({
      lineWidth: 2,
      ...options
    });
    
    series.setData(data);
    
    this.series[seriesName] = series;
    return series;
  }
  
  addMarkers(seriesName, markers) {
    if (!this.chart || !this.series[seriesName]) return;
    
    this.markers = markers;
    this.series[seriesName].setMarkers(markers);
  }
  
  clearMarkers() {
    if (!this.chart) return;
    
    // Clear markers from all series
    for (const seriesName in this.series) {
      if (this.series[seriesName].setMarkers) {
        this.series[seriesName].setMarkers([]);
      }
    }
    
    this.markers = [];
  }
  
  updateData(seriesName, data) {
    if (!this.chart || !this.series[seriesName]) return;
    
    if (seriesName === 'candlestick') {
      this.candleData = data;
      this.updateLegend(data[data.length - 1]);
    }
    
    this.series[seriesName].update(data);
  }
  
  fitContent() {
    if (!this.chart) return;
    
    this.chart.timeScale().fitContent();
  }
  
  dispose() {
    if (!this.chart) return;
    
    if (this.resizeHandler) {
      window.removeEventListener('resize', this.resizeHandler);
    }
    
    this.chart.remove();
    this.chart = null;
    this.series = {};
    this.markers = [];
  }
}

// Export visualization components
export {
  TokenRelationshipGraph,
  TokenometricsDashboard,
  TradingViewChartManager
};

// Example usage:
/*
// D3.js Network Graph
const relationshipGraph = new TokenRelationshipGraph('token-network-container');
relationshipGraph.initialize();
relationshipGraph.render(nodes, links);

// ECharts Dashboard
const dashboard = new TokenometricsDashboard();
dashboard.initialize('dashboard-container');
dashboard.updateCharts(tokenData);

// TradingView Chart
const chart = new TradingViewChartManager('tradingview-container');
chart.initialize();
chart.addCandlestickSeries(candleData);
chart.addVolumeSeries(volumeData);
*/
