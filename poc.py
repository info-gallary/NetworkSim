import streamlit as st
import os
from dotenv import load_dotenv
from agno.agent import Agent, RunOutput
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
import networkx as nx
import matplotlib.pyplot as plt
import random
import time
from PIL import Image
import io
import numpy as np

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Internet Traffic & Smart Search",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create tabs
tab1, tab2 = st.tabs(["Network Traffic Simulation", "Smart Search Agent"])

# Global variables for simulation
if 'packet_counter' not in st.session_state:
    st.session_state.packet_counter = 0
if 'total_packets' not in st.session_state:
    st.session_state.total_packets = 0
if 'delivered_packets' not in st.session_state:
    st.session_state.delivered_packets = 0
if 'total_hops' not in st.session_state:
    st.session_state.total_hops = 0
if 'packets' not in st.session_state:
    st.session_state.packets = []
if 'running' not in st.session_state:
    st.session_state.running = False

# Define a class for packet
class Packet:
    def __init__(self, source, destination, content, packet_id):
        self.source = source
        self.destination = destination
        self.content = content
        self.current_node = source
        self.path = [source]
        self.delivered = False
        self.id = packet_id
        self.color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        self.ttl = 15  # Time to live

    def move(self, next_node):
        self.current_node = next_node
        self.path.append(next_node)
        self.ttl -= 1
        if self.ttl <= 0:
            return False
        return True

# Network simulation tab content
with tab1:
    st.title("Internet Traffic Simulation")
    st.markdown("""
    This interactive simulator demonstrates how data packets travel through the internet. 
    Watch as packets find their way from source to destination through a network of routers and servers.
    Customize parameters in the sidebar to see how they affect network behavior.
    """)
    
    # Sidebar for simulation controls
    with st.sidebar:
        st.header("Simulation Controls")
        
        # Network structure
        st.subheader("Network Structure")
        num_nodes = st.slider("Number of Nodes", 10, 50, 20)
        connection_density = st.slider("Connection Density", 0.1, 0.5, 0.2)
        
        # Traffic settings
        st.subheader("Traffic Settings")
        packet_rate = st.slider("Packet Generation Rate", 1, 10, 3, help="Packets per second")
        routing_algorithm = st.selectbox("Routing Algorithm", ["Shortest Path", "Random Path"])
        simulation_speed = st.slider("Simulation Speed", 0.1, 2.0, 1.0)
        
        # Node type distribution
        st.subheader("Node Type Distribution")
        exchange_percent = st.slider("Internet Exchange Points (%)", 5, 30, 10)
        server_percent = st.slider("Servers (%)", 10, 40, 20)
        user_percent = st.slider("End Users (%)", 20, 50, 30)
        # Router percent is automatically calculated as the remainder
        
        # Packet type distribution
        st.subheader("Packet Type Distribution")
        packet_types = {
            "HTTP/Web": st.slider("HTTP/Web Traffic (%)", 10, 90, 40),
            "Video": st.slider("Video Streaming (%)", 5, 70, 30),
            "Email": st.slider("Email (%)", 5, 30, 10),
            "File Transfer": st.slider("File Transfer (%)", 5, 40, 15),
            "Other": st.slider("Other Traffic (%)", 0, 20, 5)
        }
        
        # Start/Stop buttons
        col1, col2 = st.columns(2)
        with col1:
            start_button = st.button("Start Simulation")
        with col2:
            stop_button = st.button("Stop Simulation")
            
        st.divider()
        
        # Network elements explanation
        st.subheader("Network Elements")
        st.markdown("""
        - **Blue Nodes**: Regular Routers
        - **Green Nodes**: Internet Exchange Points
        - **Red Nodes**: Servers
        - **Yellow Nodes**: End User Devices
        - **Lines**: Network Connections
        - **Colored Dots**: Data Packets
        """)

    # Main simulation display
    col1, col2 = st.columns([3, 1])
    
    with col1:
        simulation_container = st.container()
    
    with col2:
        # Network stats
        stats_container = st.container()
        
        with stats_container:
            st.subheader("Network Statistics")
            total_packets_counter = st.metric("Total Packets", "0")
            delivered_packets_counter = st.metric("Delivered Packets", "0")
            network_load = st.metric("Network Load", "0%")
            avg_delivery_time = st.metric("Avg. Delivery Time", "0 hops")
            
            # Packet visualization
            st.subheader("Active Packets")
            packet_info = st.empty()
    
    # Create the network graph
    def create_network(num_nodes, connection_density):
        G = nx.barabasi_albert_graph(num_nodes, max(1, int(num_nodes * connection_density)))
        
        # Assign node types
        node_types = {}
        exchange_count = int(num_nodes * exchange_percent / 100)
        server_count = int(num_nodes * server_percent / 100)
        user_count = int(num_nodes * user_percent / 100)
        
        nodes = list(G.nodes())
        random.shuffle(nodes)
        
        # Assign by count
        idx = 0
        for _ in range(exchange_count):
            if idx < len(nodes):
                node_types[nodes[idx]] = "exchange"
                idx += 1
        
        for _ in range(server_count):
            if idx < len(nodes):
                node_types[nodes[idx]] = "server"
                idx += 1
        
        for _ in range(user_count):
            if idx < len(nodes):
                node_types[nodes[idx]] = "user"
                idx += 1
        
        # Remaining are routers
        for i in range(idx, len(nodes)):
            node_types[nodes[i]] = "router"
        
        # Set node positions for visualization
        pos = nx.spring_layout(G, seed=42)
        return G, node_types, pos
    
    # Create network
    G, node_types, pos = create_network(num_nodes, connection_density)
    
    # Calculate shortest paths for all pairs
    shortest_paths = {}
    for source in G.nodes():
        for target in G.nodes():
            if source != target:
                try:
                    shortest_paths[(source, target)] = nx.shortest_path(G, source, target)
                except nx.NetworkXNoPath:
                    shortest_paths[(source, target)] = None
    
    # Draw the network
    def draw_network(G, node_types, pos, packets):
        plt.figure(figsize=(10, 8))
        
        # Draw the nodes with different colors based on type
        node_colors = []
        for node in G.nodes():
            if node_types[node] == "exchange":
                node_colors.append("green")
            elif node_types[node] == "server":
                node_colors.append("red")
            elif node_types[node] == "user":
                node_colors.append("yellow")
            else:
                node_colors.append("skyblue")
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        # Draw packets as small dots at their current positions
        for packet in packets:
            if not packet.delivered and packet.current_node is not None:
                plt.scatter(pos[packet.current_node][0], pos[packet.current_node][1], 
                           color=packet.color, s=100, zorder=5)
        
        plt.title("Internet Network Simulation")
        plt.axis('off')
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        # Return the image
        return Image.open(buf)
    
    # Simulate packet movement
    def route_packet(packet, G, shortest_paths):
        if packet.current_node == packet.destination or packet.delivered:
            packet.delivered = True
            return True
            
        if routing_algorithm == "Shortest Path":
            path = shortest_paths.get((packet.current_node, packet.destination))
            if path and len(path) > 1:
                next_node = path[1]  # Next node in the shortest path
                return packet.move(next_node)
        else:  # Random Path
            neighbors = list(G.neighbors(packet.current_node))
            if neighbors:
                # Prefer neighbors that lead closer to destination if possible
                if random.random() < 0.7:  # 70% chance to move closer if possible
                    better_neighbors = []
                    for n in neighbors:
                        # Check if this neighbor is on a path to destination
                        if shortest_paths.get((n, packet.destination)):
                            if len(shortest_paths.get((n, packet.destination))) < len(shortest_paths.get((packet.current_node, packet.destination), [])):
                                better_neighbors.append(n)
                    
                    if better_neighbors:
                        next_node = random.choice(better_neighbors)
                        return packet.move(next_node)
                
                # Otherwise or if no better neighbor found, choose randomly
                next_node = random.choice(neighbors)
                return packet.move(next_node)
        
        return False  # Cannot route
    
    # Generate packets - fixed version without nonlocal
    def generate_packets(G, node_types, num_packets):
        new_packets = []
        
        # Create weighted list for packet types
        packet_type_list = []
        for ptype, percent in packet_types.items():
            packet_type_list.extend([ptype] * percent)
        
        for _ in range(num_packets):
            # Prefer user nodes as sources and servers as destinations
            user_nodes = [n for n, t in node_types.items() if t == "user"]
            server_nodes = [n for n, t in node_types.items() if t == "server"]
            
            if not user_nodes or not server_nodes:
                source = random.choice(list(G.nodes()))
                destination = random.choice([n for n in G.nodes() if n != source])
            else:
                source = random.choice(user_nodes if user_nodes else list(G.nodes()))
                destination = random.choice(server_nodes if server_nodes else [n for n in G.nodes() if n != source])
            
            # Use session state instead of nonlocal
            st.session_state.packet_counter += 1
            content = random.choice(packet_type_list) if packet_type_list else "HTTP/Web"
            new_packet = Packet(source, destination, content, st.session_state.packet_counter)
            new_packets.append(new_packet)
        
        return new_packets
    
    # Function to update the animation - fixed version without nonlocal
    def update_simulation():
        if st.session_state.running:
            # Generate new packets based on packet rate
            if random.random() < packet_rate / 10:
                new_packets = generate_packets(G, node_types, random.randint(1, 3))
                st.session_state.packets.extend(new_packets)
                st.session_state.total_packets += len(new_packets)
            
            # Route each packet
            for packet in st.session_state.packets:
                if not packet.delivered:
                    delivered = route_packet(packet, G, shortest_paths)
                    if delivered and packet.current_node == packet.destination:
                        st.session_state.delivered_packets += 1
                        st.session_state.total_hops += len(packet.path) - 1  # Count hops for delivered packets
            
            # Update stats
            total_packets_counter.metric("Total Packets", f"{st.session_state.total_packets}")
            delivered_packets_counter.metric("Delivered Packets", f"{st.session_state.delivered_packets}")
            
            # Calculate network load (% of nodes with active packets)
            active_nodes = set()
            for packet in st.session_state.packets:
                if not packet.delivered:
                    active_nodes.add(packet.current_node)
            
            load_percentage = int((len(active_nodes) / len(G.nodes())) * 100)
            network_load.metric("Network Load", f"{load_percentage}%")
            
            # Calculate average delivery time in hops
            avg_hops = st.session_state.total_hops / st.session_state.delivered_packets if st.session_state.delivered_packets > 0 else 0
            avg_delivery_time.metric("Avg. Delivery Time", f"{avg_hops:.1f} hops")
            
            # Display active packet info
            active_packets = [p for p in st.session_state.packets if not p.delivered]
            if active_packets:
                packet_text = ""
                for i, p in enumerate(active_packets[:5]):  # Show top 5 packets
                    packet_text += f"**Packet #{p.id}**  \n"
                    packet_text += f"Type: {p.content}  \n"
                    packet_text += f"Path: {len(p.path)} hops  \n"
                    packet_text += f"Status: In transit  \n\n"
                if len(active_packets) > 5:
                    packet_text += f"... and {len(active_packets) - 5} more packets"
                packet_info.markdown(packet_text)
            else:
                packet_info.markdown("No active packets")
            
            # Remove packets that are delivered or cannot be routed anymore
            st.session_state.packets = [p for p in st.session_state.packets if not p.delivered and p.ttl > 0]
            
            # Generate and display the network image
            network_image = draw_network(G, node_types, pos, st.session_state.packets)
            simulation_container.image(network_image,  use_container_width=True)
            
            # Control simulation speed
            time.sleep(1.0 / simulation_speed)
    
    # Start/stop controls
    if start_button:
        st.session_state.running = True
    if stop_button:
        st.session_state.running = False
    
    # Initialize with a static image
    if 'initialized' not in st.session_state:
        network_image = draw_network(G, node_types, pos, [])
        simulation_container.image(network_image, use_container_width=True)
        st.session_state.initialized = True
    
    # Run the simulation
    if st.session_state.running:
        update_simulation()
        st.rerun()  # ✅ Use this instead of st.experimental_rerun()


# Smart Search Agent tab content
with tab2:
    st.title("Smart Search Agent")
    st.markdown("""
    This intelligent agent performs real-time searches to retrieve accurate, up-to-date information.
    It can search the web, fetch stock data, and provide structured responses to your queries.
    """)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your Smart Search Agent. I can search the web in real-time and provide you with accurate information. What would you like to know?"}
        ]
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Function to get agent response
    def get_agent_response(query):
        agent = Agent(
            model=Groq(id="llama-3.1-8b-instant"),
            description="An intelligent web search agent that performs real-time searches to retrieve the most accurate, up-to-date, and relevant information.\n"
                      "It prioritizes authoritative sources, extracts key insights, and presents structured summaries for user queries.\n"
                      "The agent ensures all responses are backed by authentic sources and provides direct links for further verification.",
            tools=[DuckDuckGoTools()],
            markdown=True
        )
        
        try:
            output: RunOutput = agent.run(query)
            return output.content
        except Exception as e:
            return f"I encountered an error while searching: {str(e)}. Please try again with a different query."
    
    # Accept user input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Show thinking indicator
        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown("Searching for information...")
            
            # Get agent response
            response = get_agent_response(prompt)
            
            # Replace thinking indicator with actual response
            thinking_placeholder.empty()
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
