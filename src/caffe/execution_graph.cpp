

#include <queue>
#include "caffe/execution_graph.hpp"

#define INF 9999999.0

using namespace std;

namespace caffe {

void ExecutionGraph::setUpExecutionGraphLayers() {
  const vector<shared_ptr<Layer<float> > >& layers = net_->layers();
  const vector<string>& layer_names = net_->layer_names();
  int num_layers = layers.size();

  int layer_id = 0;
  ExecutionGraphLayer* current_layer = NULL;
  for (int i = 0; i < num_layers; i++) {
    const string& layer_full_name = layer_names[i];
    string layer_name = layer_full_name.substr(0, layer_full_name.find("/"));

    // Skip split layers
    // Because split layers do not have useful information but give wrong output param numbers
    if (layer_full_name.find("split") != string::npos) {
      continue;
    }

    if (layer_names_index_.find(layer_name) == layer_names_index_.end()) {
      // create a new layer and set it as the current layer
      current_layer = new ExecutionGraphLayer(layer_name);
      graph_layers_.push_back(current_layer);
      layer_names_index_[layer_name] = layer_id++;
      current_layer->start_layer_id = i;

      if (i == 0) {
        // set input feature size of current layer
        // (we assume there is only one input)
        current_layer->input_feature_size = net_->input_blobs()[0]->count(1) * sizeof(float);
      }
      else {
        // set input feature size of current layer
        for (int j = 0; j < net_->bottom_vecs()[i].size(); j++) {
          current_layer->input_feature_size += net_->bottom_vecs()[i][j]->count(1) * sizeof(float);
        }
      }
    }

    // update model size
    for (int j = 0; j < layers[i]->blobs().size(); j++) {
      current_layer->model_size += layers[i]->blobs()[j]->count() * sizeof(float);
    }

    // update output feature size
    int output_f_size = 0;
    for (int j = 0; j < net_->top_vecs()[i].size(); j++) {
      output_f_size += net_->top_vecs()[i][j]->count(1) * sizeof(float);
    }
    current_layer->output_feature_size = output_f_size;

    // update execution time
    current_layer->exec_time_c += layers[i]->get_exec_time_c();
    current_layer->exec_time_s += layers[i]->get_exec_time_s();

    current_layer->end_layer_id = i;
  }
}
void ExecutionGraph::printLayers() {
  cout << "There are " << graph_layers_.size()  << " layers" << endl;
  for (int i = 0; i < graph_layers_.size(); i++) {
//    cout << graph_layers_[i]->name << endl;
    graph_layers_[i]->printExecutionGraphLayer();
  }
}

//void ExecutionGraph::printEdges(ExecutionGraph::OptTarget opt_target, int i) {
//  list<pair<int, float> >* graph = opt_target == TIME ? time_graph_ : energy_graph_;
//  cout << "Layer " << i << " " << graph_layers_[i]->name << " edges" << endl;
//
//  int start_idx = opt_target == TIME ? ((i * 4) + 1) : ((i * 3) + 1);
//  graph
//}

void ExecutionGraphLayer::printExecutionGraphLayer() {
  cout << "name : " << name
  << " input_feature: " << input_feature_size
  << " output_feature: " << output_feature_size
  << " model_size: " << model_size
  << " exec_time_c: " << exec_time_c
  << " exec_time_s: " << exec_time_s
  << endl;
}

void ExecutionGraph::addEdge(list<pair<int, float> >* graph, int src, int dst, float weight) {
  graph[src].push_back(make_pair(dst,weight));
//  cout << "(" << src << ", " << dst << ", " << weight << ")" << endl;
}

void ExecutionGraph::createTimeExecutionGraph() {
  time_graph_ = new list<pair<int, float> >[(4 * graph_layers_.size()) + 2];   // +2 for input,output

  // edge from input
  addEdge(time_graph_, 0, 1, 0.0);

  for (int i = 0; i < graph_layers_.size(); i++) {
    int idx = (4 * i) + 1;
    float input_feature_size = graph_layers_[i]->input_feature_size;
    float output_feature_size = graph_layers_[i]->output_feature_size;
    int model_size = graph_layers_[i]->model_size;
    float exec_time_c = graph_layers_[i]->exec_time_c;
    float exec_time_s = graph_layers_[i]->exec_time_s;

    // set edges inside a layer
    addEdge(time_graph_, idx, idx + 1, (static_cast<float>(model_size) + input_feature_size)/network_speed_);
    addEdge(time_graph_, idx + 1, idx + 2, exec_time_s);
    addEdge(time_graph_, idx + 2, idx + 3, output_feature_size/network_speed_);
    addEdge(time_graph_, idx, idx + 3, exec_time_c);

    // set edges in-between layers
    // client route
    addEdge(time_graph_, idx + 3, idx + 4, 0.0);

    // server route
    if (i > 0)
      addEdge(time_graph_, idx - 2, idx + 1, static_cast<float>(model_size)/network_speed_);
  }
}

void ExecutionGraph::createEnergyExecutionGraph() {
  energy_graph_ = new list<pair<int, float> >[(3 * graph_layers_.size()) + 2];   // +2 for input,output

  // edge from input
  addEdge(energy_graph_, 0, 1, 0.0);

  for (int i = 0; i < graph_layers_.size(); i++) {
    int idx = (3 * i) + 1;
    float input_feature_size = graph_layers_[i]->input_feature_size;
    float output_feature_size = graph_layers_[i]->output_feature_size;
    int model_size = graph_layers_[i]->model_size;
    float exec_time_c = graph_layers_[i]->exec_time_c;

    // set edges inside a layer
    addEdge(energy_graph_, idx, idx + 1, transfer_watt * ((static_cast<float>(model_size) + input_feature_size)/network_speed_));
    addEdge(energy_graph_, idx + 1, idx + 2, transfer_watt * (output_feature_size/network_speed_));
    addEdge(energy_graph_, idx, idx + 2, compute_watt * exec_time_c);

    // set edges in-between layers
    // client route
    addEdge(energy_graph_, idx + 2, idx + 3, 0.0);

    // server route
    if (i > 0)
      addEdge(energy_graph_, idx - 2, idx + 1, transfer_watt * (static_cast<float>(model_size)/network_speed_));
  }
}

void ExecutionGraph::getBestPathForTime(list<pair<int, int> >* result) {
  shortestPath(TIME, result);
}

void ExecutionGraph::getBestPathForEnergy(list<pair<int, int> >* result) {
  shortestPath(ENERGY, result);
}

typedef pair<float, int> fiPair;

static bool isServerNode(ExecutionGraph::OptTarget opt_target, int id) {
  if (opt_target == ExecutionGraph::TIME) {
    int offset = (id - 1) % 4;
    return (offset == 1) || (offset == 2);
  }
  else if (opt_target == ExecutionGraph::ENERGY) {
    int offset = (id - 1) % 3;
    return (offset == 1);
  }
  // unreachable
  CHECK(false);
  return false;
}

void ExecutionGraph::shortestPath(OptTarget opt_target, list<pair<int, int> >* result) {

  list<pair<int, float> >* graph = NULL;
  int V = 0;
  if (opt_target == TIME) {
    graph = time_graph_;
    V = (4 * graph_layers_.size()) + 2;
  }
  else if (opt_target == ENERGY) {
    graph = energy_graph_;
    V = (3 * graph_layers_.size()) + 2;
  }

  int src = 0;

	// Save the previous node of each node to restore the shortest path
	vector<int> path(V, -1);

	// Create a priority queue to store vertices that
	// are being preprocessed
	priority_queue< fiPair, vector <fiPair> , greater<fiPair> > pq;

	// Create a vector for distances and initialize all
	// distances as infinite (INF)
	vector<float> dist(V, INF);

	// Insert source itself in priority queue and initialize
	// its distance as 0.
	pq.push(make_pair(0.0, src));
	dist[src] = 0.0;

	/* Looping till priority queue becomes empty (or all
		distances are not finalized) */
	while (!pq.empty())	{
		// The first vertex in pair is the minimum distance
		// vertex, extract it from priority queue.
		int u = pq.top().second;
		pq.pop();

		// 'i' is used to get all adjacent vertices of a vertex
		list< pair<int, float> >::iterator i;
		for (i = graph[u].begin(); i != graph[u].end(); ++i) {
			// Get vertex label and weight of current adjacent of u
			int v = (*i).first;
			int weight = (*i).second;

			//  If there is shorted path to v through u.
			if (dist[v] > dist[u] + weight) {
				// Updating distance of v
				dist[v] = dist[u] + weight;
				pq.push(make_pair(dist[v], v));
				path[v] = u;
			}
		}
	}

	// Print shortest distances stored in dist[]
//	cout << "Vertex   Distance from Source" << endl;
//	for (int i = 0; i < V; ++i)
//			cout << i << "\t\t" << dist[i] << endl;

	cout << "Shortest path from src to dst" << endl;
	int node = V-1;
  int resume_node = 0;
	while (node != src) {
    if (!isServerNode(opt_target, node) && isServerNode(opt_target, path[node])) {
      resume_node = node;
    }
    else if (isServerNode(opt_target, node) && !isServerNode(opt_target, path[node])) {
      // get index of real caffe layers
      int offloading_point = graph_layers_[(path[node] - 1)/4]->start_layer_id;
      int resume_point = graph_layers_[(resume_node - 1)/4]->end_layer_id;

      result->push_front(make_pair(offloading_point, resume_point));

      // Logging for time opt (comparison beween local execution and offloading)
      cout << "offload (" << offloading_point << ", " << resume_point << ") ";
      float c_time = 0.0;
      float s_time = 0.0;
      int s = (path[node]-1)/4;
      int e = (resume_node-1)/4;
      for (int idx = s; idx <= e; idx++) {
        c_time += graph_layers_[idx]->exec_time_c;
        int model_size = graph_layers_[idx]->model_size;
        float input_feature_size = graph_layers_[idx]->input_feature_size;
        float output_feature_size = graph_layers_[idx]->output_feature_size;
        if (idx == s) {
          s_time += input_feature_size/network_speed_;
        }
        if (idx == e) {
          s_time += output_feature_size/network_speed_;
        }
        s_time += static_cast<float>(model_size)/network_speed_;
        s_time += graph_layers_[idx]->exec_time_s;
      }
      cout << "local time : " << c_time << ", offload time : " << s_time << endl;
    }

		cout << node << " ";
		node = path[node];
	}
	cout << node << endl;
}

}  // namespace caffe
