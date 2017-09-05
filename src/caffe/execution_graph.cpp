

#include <queue>
#include <algorithm>
#include "caffe/execution_graph.hpp"

#define INF 9999999.0

using namespace std;

namespace caffe {

static vector<int> intersection(const vector<vector<int>* > *vecs) {
	vector<int> last_intersection = *(vecs->at(0));
	vector<int> curr_intersection;

	for (size_t i = 1; i < vecs->size(); ++i) {
		set_intersection(last_intersection.begin(), last_intersection.end(),
			vecs->at(i)->begin(), vecs->at(i)->end(),
			back_inserter(curr_intersection));
		swap(last_intersection, curr_intersection);
		curr_intersection.clear();
	}
	return last_intersection;
}

void ExecutionGraph::computeDominatorLayers() {
  const vector<shared_ptr<Layer<float> > >& layers = net_->layers();
  int num_layers = layers.size();
  int output_layer = num_layers - 1;

  // Set initial value
  for (int i = 0; i < num_layers; i++) {
    vector<int>* dominator = layers[i]->dominator();
    if (i == 0)
      dominator->push_back(0);
    else {
      for(int j = 0; j < num_layers; j++) {
        dominator->push_back(j);
      }
    }
  }

  // Get dominators
  bool changed;
  do {
    changed = false;
    for (int i = 1; i < num_layers; i++) {
      // Get intersection of parents' dom + current layer
      int num_parents = layers[i]->layer_param().bottom_size();
			vector<int> parents_layers;
      for (int j = 0; j < num_parents; j++) {
        string bottom_blob = layers[i]->layer_param().bottom(j);
        if (bottom_blob.find("split") != string::npos) {
          // This is for special characteristic of 'split' layer
          // hard coding.. but simple
          bottom_blob = bottom_blob.substr(0, bottom_blob.length() - 2);
        }
        int layer_id = net_->layer_id_by_name(bottom_blob);
        if (layer_id == -1 && bottom_blob.find("data") != string::npos) {
          // This is for special characteristic of 'data' layer
					layer_id = 0;
				}
				CHECK_GE(layer_id, 0);
        parents_layers.push_back(layer_id);
      }
      vector<vector<int>* > parents_doms;
      for (int j = 0; j < parents_layers.size(); j++) {
				parents_doms.push_back(layers[parents_layers[j]]->dominator());
			}
			vector<int> result = intersection(&parents_doms);
    	result.push_back(i);

			// Compare the result with the original dom list
			// If changed, update dom list
			vector<int> v;
			vector<int>* curr_dom = layers[i]->dominator();
			set_difference(curr_dom->begin(), curr_dom->end(), result.begin(), result.end(), inserter(v, v.begin()));
			if (v.size() > 0) {
				changed = true;
				curr_dom->assign(result.begin(), result.end());
			}
		}
  } while (changed);

	// Print dominators of output layer
	cout << "Dominators of output layer:";
	vector<int>* output_dom = layers[output_layer]->dominator();
  // remove output layer
  output_dom->pop_back();

  // i = 1 -> remove input layer
  for (int i = 1; i < output_dom->size(); i++) {
    if (strcmp(layers[output_dom->at(i)]->type(), "Split") != 0) {
      dominators_.push_back(output_dom->at(i));
      cout << " " << output_dom->at(i);
	  }
  }
	cout << endl;

	cout << "# of Dominators: " << dominators_.size() << endl;

  // Clear dominators
  // (This should be chagned.
  // we can allocate dominators in this function as local variables
  // so that they can be cleared naturally)
  for (int i = 0; i < num_layers; i++) {
    layers[i]->dominator()->clear();
  }
}

void ExecutionGraph::setUpExecutionGraphLayers() {
  const vector<shared_ptr<Layer<float> > >& layers = net_->layers();
  const vector<string>& layer_names = net_->layer_names();
  int num_layers = layers.size();

  ExecutionGraphLayer* current_layer = NULL;

  // ID of next dominator
  int next_dom_idx = dominators_.size() - 1;
  int next_dom = dominators_[next_dom_idx];

  // Create layers from output to input layer
  for (int i = num_layers - 2; i > 0 ; i--) {
    // Skip split layers
    // Because split layers do not have useful information but give wrong output param numbers
    if (strcmp(layers[i]->type(), "Split") == 0) {
      continue;
    }

    if (i == next_dom) {
      const string& layer_name = layer_names[i];

      // Create a new execution graph layer
      current_layer = new ExecutionGraphLayer(layer_name);
      current_layer->end_layer_id = i;

      // set output feature size of current layer
      for (int j = 0; j < net_->top_vecs()[i].size(); j++) {
        current_layer->output_feature_size += net_->top_vecs()[i][j]->count(1) * sizeof(float);
      }

			// push front
		  vector<ExecutionGraphLayer*>::iterator it = graph_layers_.begin();
			it = graph_layers_.insert (it , current_layer);

      // update next_dom (take care of out-of-range case)
      if (--next_dom_idx >= 0)
        next_dom = dominators_[next_dom_idx];
    }

    current_layer->start_layer_id = i;

    // update model size
    for (int j = 0; j < layers[i]->blobs().size(); j++) {
      current_layer->model_size += layers[i]->blobs()[j]->count() * sizeof(float);
    }

    // update input feature size
    int input_f_size = 0;
    for (int j = 0; j < net_->bottom_vecs()[i].size(); j++) {
      input_f_size += net_->bottom_vecs()[i][j]->count(1) * sizeof(float);
    }
    current_layer->input_feature_size = input_f_size;

    // update execution time
    current_layer->exec_time_c += layers[i]->get_exec_time_c();
    current_layer->exec_time_s += layers[i]->get_exec_time_s();

	// prototxt size
	NetParameter layer_prototxt_param;
	net_->ToProtoNoBlob(&layer_prototxt_param, false, current_layer->start_layer_id, current_layer->end_layer_id);
	current_layer->prototxt_size = layer_prototxt_param.ByteSize();

	// server-side loading time
	current_layer->loading_time_s = 2.6357798e-3 * current_layer->prototxt_size + 10.8014505382;
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
  << " prototxt_size: " << prototxt_size
  << " loading_time_s: " << loading_time_s
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
    float loading_time_s = graph_layers_[i]->loading_time_s;

    // set edges inside a layer
    addEdge(time_graph_, idx, idx + 1, (static_cast<float>(model_size) + input_feature_size)/network_speed_);
    addEdge(time_graph_, idx + 1, idx + 2, exec_time_s);
    addEdge(time_graph_, idx + 2, idx + 3, output_feature_size/network_speed_);
    addEdge(time_graph_, idx, idx + 3, exec_time_c);

    // set edges in-between layers
    // client route
    addEdge(time_graph_, idx + 3, idx + 4, 0.0);

    // server route
    if (i > 0){
      addEdge(time_graph_, idx - 2, idx + 1, static_cast<float>(model_size)/network_speed_);
    }
  }
}

// model transfer cost is updated to (1 - k) * model_transfer_cost
// ex1 : k = 0, model transfer cost is the same with first offloading
// ex2 : k = 1, model transfer cost will be 0
void ExecutionGraph::updateTimeExecutionGraphWeight(float k) {
  for (int i = 0; i < graph_layers_.size(); i++) {
    int idx = (4 * i) + 1;
    float input_feature_size = graph_layers_[i]->input_feature_size;
    int model_size = graph_layers_[i]->model_size;

    // update edege weights for transmitting a DNN model
		list< pair<int, float> >::iterator j;
		for (j = time_graph_[idx].begin(); j != time_graph_[idx].end(); ++j) {
			int v = (*j).first;
      if (v == idx + 1)
        (*j).second = ((static_cast<float>(model_size)*(1-k)) + input_feature_size)/network_speed_;
    }

    if (i > 0){
      // update edege weights for transmitting a DNN model
      for (j = time_graph_[idx - 2].begin(); j != time_graph_[idx - 2].end(); ++j) {
        int v = (*j).first;
        if (v == idx + 1)
          (*j).second = (static_cast<float>(model_size)*(1-k))/network_speed_;
      }
    }
  }
}

void ExecutionGraph::updateEnergyExecutionGraphWeight(float k) {
  for (int i = 0; i < graph_layers_.size(); i++) {
    int idx = (3 * i) + 1;
    float input_feature_size = graph_layers_[i]->input_feature_size;
    int model_size = graph_layers_[i]->model_size;

    // update edege weights for transmitting a DNN model
		list< pair<int, float> >::iterator j;
		for (j = energy_graph_[idx].begin(); j != energy_graph_[idx].end(); ++j) {
			int v = (*j).first;
      if (v == idx + 1)
        (*j).second = (transfer_watt * ((static_cast<float>(model_size)*(1-k)) + input_feature_size))/network_speed_;
    }

    if (i > 0){
      // update edege weights for transmitting a DNN model
      for (j = energy_graph_[idx - 2].begin(); j != energy_graph_[idx - 2].end(); ++j) {
        int v = (*j).first;
        if (v == idx + 1)
          (*j).second = (transfer_watt * (static_cast<float>(model_size)*(1-k)))/network_speed_;
      }
    }
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
	bool result;
	int offset;
	switch(opt_target) {
		case ExecutionGraph::TIME:
			offset = (id - 1) % 4;
			result = (offset == 1) || (offset == 2);
			break;
		case ExecutionGraph::ENERGY:
			offset = (id - 1) % 3;
			result = (offset == 1);
			break;
		default:
			// unreachable
			CHECK(false);
	}
	return result;
}

void ExecutionGraph::shortestPath(OptTarget opt_target, list<pair<int, int> >* result) {

  list<pair<int, float> >* graph = NULL;
  int V = 0;	// # of nodes

	switch(opt_target) {
		case TIME:
			graph = time_graph_;
			V = (4 * graph_layers_.size()) + 2;
			break;
		case ENERGY:
			graph = energy_graph_;
			V = (3 * graph_layers_.size()) + 2;
			break;
		default:
			// unreachable
			CHECK(false);
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
			float weight = (*i).second;

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

			int nodes_per_layer = opt_target == TIME ? 4 : 3;

      // get index of real caffe layers
      int offloading_point = graph_layers_[(path[node] - 1)/nodes_per_layer]->start_layer_id;
      int resume_point = graph_layers_[(resume_node - 1)/nodes_per_layer]->end_layer_id;

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
        s_time += graph_layers_[idx]->loading_time_s;
      }
      cout << "local time : " << c_time << ", offload time : " << s_time << endl;
    }

		cout << node << " ";
		node = path[node];
	}
	cout << node << endl;
}

}  // namespace caffe
