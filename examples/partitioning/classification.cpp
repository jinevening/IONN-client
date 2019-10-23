#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <termios.h>

#define BUFF_SIZE 1024*1024*256

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;
boost::condition_variable_any cond;
boost::mutex mutex;
boost::mutex mutex2;
int total_uploaded_feature = 0;
int total_downloaded_feature = 0;
int total_uploaded_model = 0;
struct timeval start_global;

class Classifier {
 public:
  Classifier(const string& model_file,
             const string& trained_file,
             const string& mean_file,
             const string& label_file,
             const string& client_prediction_model_file,
             const string& server_prediction_model_file,
             float network_speed,
             tcp::socket* s,
             float K,
             bool incremental,
             const string& opt_target,
             bool use_file,
             const string& ip_address);

  void Classify(const cv::Mat& img, int N = 5);
  void StartBackgroundUpload();
  void JoinBackgroundUpload();
  void StartClassify(cv::Mat& img);
  void JoinStartClassify();
  vector<bool> upload_complete_;

 private:
  // For setting up prediction models
  void ServerPredict(const string& server_prediction_model_file, bool use_file);
  void ClientPredict(const string& client_prediction_model_file);

  void SetMean(const string& mean_file);

  std::vector<float> Predict(const cv::Mat& img);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

  void IncrementalUploading();
  void IncrementalOffloadingSingleIteration(int plan_index, list< pair<pair<int, int>, pair<int, int> > > &to_offload_layers);
  
  void SequentialClassifing(  cv::Mat& img );

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<string> labels_;
  shared_ptr<ExecutionGraph> graph_;
  int num_offloading_;
  float K_;  // model tranfer cost decreasing rate
  bool incremental_;
  boost::thread upload_thread_;
  boost::thread classify_thread_;
  vector<list<pair<int, int> >* > partitioning_plans_;
  list<pair<int, int> > offloaded_layers_;
  ExecutionGraph::OptTarget opt_target_;
  string ip_address_;
};

void Classifier::StartBackgroundUpload() {
  upload_thread_ = boost::thread(boost::bind(&Classifier::IncrementalUploading, this));
};

void Classifier::JoinBackgroundUpload() {
  upload_thread_.join();
};

void Classifier::StartClassify(cv::Mat& img) {
  classify_thread_ = boost::thread(boost::bind(&Classifier::SequentialClassifing, this, img));
};

void Classifier::JoinStartClassify() {
  classify_thread_.join();
};

void Classifier::IncrementalOffloadingSingleIteration(int plan_index, list< pair<pair<int, int>, pair<int, int> > > &to_offload_layers)
{
  to_offload_layers.clear();

  if(plan_index >= partitioning_plans_.size()){
    cout<<"Error, planindex out of range";
    return;
  }
  if(plan_index == 0){
    /* Nothing has been offloaded. return every layer as a front model */
    for(list<pair<int, int> >::iterator j = partitioning_plans_[0]->begin(); j != partitioning_plans_[0]->end(); j++){
      to_offload_layers.push_back(make_pair(make_pair(j->first, j->second), make_pair(-1,-1)));
    }
    return;
  }
  else {
    list<pair<int, int> >* plan_aim = partitioning_plans_[plan_index];
    list<pair<int, int> >* plan_offloaded = partitioning_plans_[plan_index-1];
    list<pair<int, int> >::iterator j = plan_aim->begin();
    list<pair<int, int> >::iterator k = plan_offloaded->begin();
    int start = j->first;
    int end = j->second;
    while(j != plan_aim->end()){
      bool incrementj = false;
      bool incrementk = false;

      if(k->first < start){
        cout << "Error, this plan[-1] should have been processed" << endl;
      }
      
      int offloaded_start;
      int offloaded_end;
      int rear_second_limit = -1; 

      int front_first;
      int front_second;
      int rear_first;
      int rear_second;
      bool k_within_j;
      if(k==plan_offloaded->end() or k->first > end){
        offloaded_start = -1;
        offloaded_end = -1;
        k_within_j = false;
      }
      else{
        offloaded_start = k->first;
        offloaded_end = k->second;
        k_within_j = true;
     }
      if(k_within_j == true){
        list<pair<int, int> >::iterator l = k;
        l++;
       if(l != plan_offloaded->end()){
          rear_second_limit = l->first - 1;
       }
      }

      if(k_within_j == false or start < offloaded_start){
        //have to make front layer, and this layer will have a additional input layer added
        front_first = start;
        if(k_within_j == false){
          front_second = end;
          incrementj = true;
       }
        else{
          front_second = offloaded_start - 1;
          start = offloaded_start;
       }
      }
      else{ // start == offloaded_start
        front_first = -1;
        front_second = -1;
     }
      if(k_within_j){
        incrementk = true; // since k is in j so k is should be processed
     }

      //front_* obatined, next get rear_*
      if(k_within_j == false or end == offloaded_end){
        rear_first = -1;
        rear_second = -1;
        incrementj = true;
     }
      else { //end > offloaded_end
        if(rear_second_limit == end){ // if next k starts at end+1
          cout << "Error, should be unreachable";
       }
        if(rear_second_limit <= -1 or rear_second_limit > end){
          rear_first = offloaded_end + 1;
          rear_second = end;
          incrementj = true;
       }
        else{
          rear_first = offloaded_end + 1;
          rear_second = rear_second_limit;
          start = rear_second_limit+1;
          incrementj = false;
       }
      }
      if(front_first != -1 or rear_first != -1){
        if(front_second == -1 and rear_second == -1){
          cout<<"Error";
       }
        to_offload_layers.push_back(make_pair(make_pair(front_first, front_second), make_pair(rear_first, rear_second)));
    }

      if(incrementj == true){
        j++;
        if(j != plan_aim->end()){
          start= j->first;
          end = j->second;
       }
      }
      if(incrementk == true){
        k++;
     }    
    }
    if(k != plan_offloaded->end()){
      cout << "Error, prev plan not completely processed";
    }
  }
  return;
}

void Classifier::SequentialClassifing(cv::Mat& img) {

  boost::unique_lock<boost::mutex> lock = boost::unique_lock<boost::mutex>(mutex);
  cond.wait(mutex);

  cout <<"forward thread woken"<<endl;
  while(!this->upload_complete_.back()){
//    usleep(500000);
    this->Classify(img);
    //n++;
  }
  // 10 more queries after everything is uploaded
  cout << "during ionn " << total_uploaded_feature << " bytes data uploaded " 
  << total_downloaded_feature << " bytes data downloaded" <<endl;


  for (int i = 0; i < 10; i++) {
//    usleep(500000);
    this->Classify(img);
  }

  double timechk;
  double timechk2;
	struct timeval start;
	struct timeval finish;
	gettimeofday(&start, NULL);




//  for (int i = 0; i < 100; i++) {
////    usleep(500000);
//    this->Classify(img);
//  }
//  gettimeofday(&finish, NULL);
//  timechk = (double)(finish.tv_sec) + (double)(finish.tv_usec) / 1000000.0 -
//            (double)(start.tv_sec) - (double)(start.tv_usec) / 1000000.0;
//
//  cout << "Average optimal query time : " << timechk/100 << " s" << endl;

}

void Classifier::IncrementalUploading() {
  boost::asio::io_service io_service;
  tcp::socket s(io_service);
  tcp::resolver resolver(io_service);
  tcp::resolver::query query(tcp::v4(), ip_address_, "7675");
  tcp::resolver::iterator iter = resolver.resolve(query);
  boost::asio::connect(s, iter);

  boost::unique_lock<boost::mutex> lock = boost::unique_lock<boost::mutex>(mutex2);
  cond.wait(mutex2);

  std::ofstream uploadFinishFile("upload_finish_time.csv", std::ios::app);
  cout <<"upload woken" << endl;

	/* Time check variables*/
  double timechk;
  double timechk2;
	struct timeval start;
	struct timeval finish;
	gettimeofday(&start, NULL);

  const bool outputlog = true;

  /* Buffer for data transmission */
  unsigned char* buff = new unsigned char[BUFF_SIZE];
  memset(buff, 0, sizeof(BUFF_SIZE));

  typedef list< pair<pair<int, int>, pair<int, int> > > listppp;
  listppp result;

  for (int i = 0; i < partitioning_plans_.size(); ++i) {
    list<pair<int, int> >* curr = partitioning_plans_[i];
//    CHECK_EQ(curr->size(), 1); 
//  for (int n = 0; n < partitioning_plans_.size(); ++n) {
//    cout << "partitioning_plans_"<<n<<" ";
//    for(list<pair<int, int> >::iterator m = partitioning_plans_[n]->begin(); m != partitioning_plans_[n]->end(); m++){
//      cout <<m->first << " " << m->second << " and ";
//    }
//    cout << endl;
//  }


    /* Get NN layers to-be-offloaded (exclude layers already offloaded to the server) */
    IncrementalOffloadingSingleIteration(i, result);
    for(listppp::iterator j = result.begin(); j != result.end(); j++){
      if(outputlog){
        cout <<"plan aim:";
        for(list<pair<int, int> >::iterator l = curr->begin(); l != curr->end(); l++){
           cout <<l->first<<" "<< l->second;
         }
        cout<<endl<< "SingleIteration"<< distance(result.begin(), j)<<": " << j->first.first <<" "<< j->first.second <<" "<< j->second.first <<" "<< j->second.second << endl;
      }
      int offloading_point;
      int resume_point;
      int prototxt_end;
      int prototxt_size = 0;
      int front_model_size = 0;
      int rear_model_size = 0;
      int total_size = 0;
      CHECK_EQ(sizeof(int), 4);

      pair<int, int> front_net = j->first;
      pair<int, int> rear_net = j->second;

/*
      memcpy(buff, &total_prototxt_size, 4);
      total_prototxt_param.SerializeWithCachedSizesToArray(buff+4);
      buff += 4+total_prototxt_size;
*/
      int found = false;
      for(list<pair<int, int> >::iterator i = curr->begin(); i != curr->end(); i++){
        int lowest, highest;
        if(j->first.first != -1){
          lowest = j->first.first;
        }
        else{
          lowest = j->second.first;
          if(j->second.first == -1){
            cout <<"Error classification.cpp lno294";
          }
        }

        if(j->second.second != -1){
          highest = j->second.second;
        }
        else{
          highest = j->first.second;
          if(j->first.second == -1){
            cout <<"Error classification.cpp lno294";
          }
        }
        if(lowest >= i->first and highest <= i->second and found == false){
          found =true;
          offloading_point = i->first;
          resume_point = i->second;
        }
        if(i->second > prototxt_size){
          prototxt_end = i->second;
        }
      }

      cout << "complete chunk: " << offloading_point<< " " << resume_point <<", prototxt: 1 " << prototxt_end<<endl;

      int sizeofvariousindexes = 28;
      /* Serialize prototxt */
      if (front_net.first != -1 || rear_net.first != -1) {
        NetParameter prototxt_param;
//        if(offloading_point == 1){
//          net_->ToProtoNoBlob(&prototxt_param, false, offloading_point, prototxt_end, true);
//        }
//        else{
        net_->ToProtoNoBlob(&prototxt_param, false, offloading_point, prototxt_end, true);
//        }
        prototxt_size = prototxt_param.ByteSize();
        prototxt_param.SerializeWithCachedSizesToArray(buff + sizeofvariousindexes);
//        cout << "prototxt size : " << prototxt_size << " bytes" << endl;
      }
      else {
        prototxt_size = 0;
      }

      /* Serialize front model blobs */
      if (front_net.first != -1) {
        NetParameter front_net_param;
        if(offloading_point == 1){
          net_->ToProto(&front_net_param, false, front_net.first, front_net.second, true);
        }
        else{
          net_->ToProto(&front_net_param, false, front_net.first, front_net.second, true);
        }
        front_model_size = front_net_param.ByteSize();
        front_net_param.SerializeWithCachedSizesToArray(buff + sizeofvariousindexes + prototxt_size);
//        cout << "Front model blobs size : " << front_model_size << " bytes" << endl;
      }
      else {
        front_model_size = 0;
//        cout << "Front model blobs size : 0 bytes" << endl;
      }

      /* Serialize rear model blobs */
      if (rear_net.first != -1) {
        NetParameter rear_net_param;
        net_->ToProto(&rear_net_param, false, rear_net.first, rear_net.second, false);
        rear_model_size = rear_net_param.ByteSize();
        rear_net_param.SerializeWithCachedSizesToArray(buff + sizeofvariousindexes + prototxt_size + front_model_size);
//        cout << "Rear model blobs size : " << rear_model_size << " bytes" << endl;
      }
      else {
        rear_model_size = 0;
//        cout << "Rear model blobs size : 0 bytes" << endl;
      }

      total_size = prototxt_size + front_model_size + rear_model_size;

      /* Header (size of transmitted data)*/
      memcpy(buff, &total_size, 4);
      memcpy(buff + 4, &prototxt_size, 4);
      memcpy(buff + 8, &front_model_size, 4);
      memcpy(buff + 12, &rear_model_size, 4);
      memcpy(buff + 16, &offloading_point, 4);
      memcpy(buff + 20, &resume_point, 4);
      memcpy(buff + 24, &prototxt_end, 4);
//      buff -= 4+total_prototxt_size;


      /* Upload DNN layers in the current plan */
//			boost::asio::write(s, boost::asio::buffer(buff, total_size + 16+4+total_prototxt_size), boost::asio::transfer_all());
			boost::asio::write(s, boost::asio::buffer(buff, total_size + sizeofvariousindexes), boost::asio::transfer_all());
      total_uploaded_model += total_size;



    	struct timeval start2;
    	struct timeval finish2;


//stack overflow, "what means blocking for boost::asio::write"
//      gettimeofday(&start2, NULL);
//      tcdrain(s.native_handle());
//      gettimeofday(&finish2, NULL);
//      timechk = (double)(finish2.tv_sec) + (double)(finish2.tv_usec) / 1000000.0 -
//            (double)(start2.tv_sec) - (double)(start2.tv_usec) / 1000000.0;
//      cout <<"tcdrain took " << timechk << " s" << endl;

      /* Wait to receive ACK from server */
//      size_t ack = s.read_some(boost::asio::buffer(buff, 100));
//      CHECK_EQ(ack, 3);
//      cout << "Received ACK" << endl;

      /* Update offloaded layers */
      offloaded_layers_.push_back(make_pair(offloading_point, resume_point));
    }
    upload_complete_[i] = true;
  }
  int no_more_partition = 0;
  memcpy(buff, &no_more_partition, 4);
  boost::asio::write(s, boost::asio::buffer(buff, 4), boost::asio::transfer_all());

  gettimeofday(&finish, NULL);
  timechk = (double)(finish.tv_sec) + (double)(finish.tv_usec) / 1000000.0 -
            (double)(start.tv_sec) - (double)(start.tv_usec) / 1000000.0;
  timechk2 = (double)(finish.tv_sec) + (double)(finish.tv_usec) / 1000000.0 -
            (double)(start_global.tv_sec) - (double)(start_global.tv_usec) / 1000000.0;

  uploadFinishFile << timechk << endl;
  cout << "Upload of "<< total_uploaded_model << " bytes of model took: " << timechk << "s at time: "<<timechk2 <<"s." << endl;
};

static bool same_list(list<pair<int, int> >* a, list<pair<int, int> >* b) {
  if (a->size() != b->size())
    return false;

  list<pair<int, int> >::iterator i = a->begin();
  list<pair<int, int> >::iterator j = b->begin();

  while(i != a->end()) {
    if ((*i).first != (*j).first || (*i).second != (*j).second)
      return false;
    i++;
    j++;
  }

  return true;
}

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const string& label_file,
                       const string& client_prediction_model_file,
                       const string& server_prediction_model_file,
                       float network_speed,
                       tcp::socket* s,
                       float K,
                       bool incremental,
                       const string& opt_target,
                       bool use_file,
                       const string& ip_address) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  ip_address_ = ip_address;

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  /* Set socket */
  net_->SetSocket(s);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file);

  /* Load labels. */
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));

  Blob<float>* output_layer = net_->output_blobs()[0];
  CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";

  /* Load prediction models */
  ServerPredict(server_prediction_model_file, use_file);
  ClientPredict(client_prediction_model_file);

  /* Create execution graph */
  graph_.reset(new ExecutionGraph(net_.get(), network_speed));
  graph_->printLayers();

  if (opt_target == "time")
    opt_target_ = ExecutionGraph::TIME;
  else if (opt_target == "energy")
    opt_target_ = ExecutionGraph::ENERGY;

  graph_->createNNExecutionGraph(opt_target_);

  /* Initilaize # of offloading */
  num_offloading_ = 0;
  K_ = K;
  incremental_ = incremental;
  offloaded_layers_.push_back(make_pair(-1, -1));
//  offloaded_layers_ = make_pair(-1,-1);
  vector<list<pair<int, int> >* > partitioning_plans_old;

 	/* Time check variables*/
  double timechk;
	struct timeval start;
	struct timeval finish;
	gettimeofday(&start, NULL);

  const bool original_algorithm = true; //true if want use original algorithm, false if efficiency algorithm

  int level = 0;
  float k;
  while (true) {
    k = 1 - pow(K_, level);
//    k = 0.0001*level;
    if (k > 0.999)
      k = 1.0;

    // Get partitioning points for minimizing execution time 
    graph_->updateNNExecutionGraphWeight(k, opt_target_);

    // Get partitioning points of best path 
    list<pair<int, int> >* partitioning_plan = new list<pair<int, int> >;
    graph_->getBestPartitioningPlanOld(partitioning_plan, opt_target_);
    
    if (!partitioning_plan->empty() &&
        (partitioning_plans_old.empty() ||
        !same_list(partitioning_plan, partitioning_plans_old.back()))) {
      partitioning_plans_old.push_back(partitioning_plan);

      //uncomment for original algorithm
      if (original_algorithm) {
        upload_complete_.push_back(false);
      }

      cout << k<<",";
    }

    level++;
  
    if (k >= 1.0)
      break;
  }

  gettimeofday(&finish, NULL);
  timechk = (double)(finish.tv_sec) + (double)(finish.tv_usec) / 1000000.0 -
            (double)(start.tv_sec) - (double)(start.tv_usec) / 1000000.0;

  cout << "Partitioning previous time taken: " << timechk << " s" << endl;

  cout << "Partitioning plans old" << endl;
  for (int i = 0; i < partitioning_plans_old.size(); ++i) {
    list<pair<int, int> >* curr = partitioning_plans_old[i];
    list<pair<int, int> >::iterator j;
    for (j = (*curr).begin(); j != (*curr).end(); ++j) {
      if(j->first != -1){
        cout << i << ". (" << (*j).first << ", " << (*j).second << ") " << endl;
      }
    }
      cout << "->";
  }

	gettimeofday(&start, NULL);
  level = 0;
  list<pair<int, int> >* partitioning_plan = new list<pair<int, int> >;
  graph_->getBestPartitioningPlan(partitioning_plan, opt_target_);
  for(list<pair<int, int> >::iterator it = (*partitioning_plan).begin(); it != (*partitioning_plan).end(); ++it){
    list<pair<int, int> >* temp = new list<pair<int, int> >;
//  list<pair<int, int> >::iterator jt = partitioning_plan->begin();
//    while(true){
//      temp->push_back((*jt));
//      if(jt == it){
//        break;
//      }
//      jt++;
//    }
    for(; it != partitioning_plan->end() and it->first != -1; ++it){
      temp->push_back((*it));
    }
//    if(it != partitioning_plan->end()){
//      it++;
//    }
    partitioning_plans_.push_back(temp);

//  comment for original algorithm
    if (!original_algorithm) {
      upload_complete_.push_back(false);
    }

    level++;
  }
  gettimeofday(&finish, NULL);
  timechk = (double)(finish.tv_sec) + (double)(finish.tv_usec) / 1000000.0 -
            (double)(start.tv_sec) - (double)(start.tv_usec) / 1000000.0;

  cout << "Partitioning new time taken: " << timechk << " s" << endl;

  /* Print partitioning plans */
  cout << "Partitioning plans" << endl;
  for (int i = 0; i < partitioning_plans_.size(); ++i) {
    list<pair<int, int> >* curr = partitioning_plans_[i];
    list<pair<int, int> >::iterator j;
    for (j = (*curr).begin(); j != (*curr).end(); ++j) {
      if(j->first != -1){
        cout << i << ". (" << (*j).first << ", " << (*j).second << ") " << endl;
      }
    }
    cout << "->";
  }

//uncomment for original algorithm
  if (original_algorithm) {
    partitioning_plans_ = partitioning_plans_old;
    cout << "using old plans" << endl;
  }
}

// Server Prediction Model
void Classifier::ServerPredict(const string& server_prediction_model_file, bool use_file){
  if (use_file) {
    net_->server_predict_from_profile(server_prediction_model_file);
  }
  else {
    if(!net_->server_predict(server_prediction_model_file)){
        CHECK(false);
        exit(0);
    }
  }
}

// Client side prediction model 
void Classifier::ClientPredict(const string& client_prediction_model_file){
	net_->client_predict(client_prediction_model_file);
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the top N predictions. */
void Classifier::Classify(const cv::Mat& img, int N) {
  cout << "[[[Offloading " << num_offloading_ << "]]]" << endl;
  std::vector<float> output = Predict(img);

  N = std::min<int>(labels_.size(), N);
  std::vector<int> maxN = Argmax(output, N);
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[idx], output[idx]));
  }

  num_offloading_++;

  /* Print the top N predictions. */
  for (size_t i = 0; i < predictions.size(); ++i) {
    Prediction p = predictions[i];
    std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
              << p.first << "\"" << std::endl;
  }
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
	/* Time check variables*/
  double timechk;
  double timechk2;
	struct timeval start;
	struct timeval finish;

  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  std::ofstream timeFile("time_measurement.csv", std::ios::app);
  std::ofstream layerFile("layer_measurement.csv", std::ios::app);
  std::ofstream forwardTimeFile("forward_time.csv", std::ios::app);

  int i = 0;
  /* Check DNN layers uploaded so far */
  for (i = 0; i < upload_complete_.size(); i++ ) {
    if (!upload_complete_[i]) {
      break;
    }
  }
  i = i - 1;

  list<pair<int, int> >* partitioning_plan;
  if (i >= 0) {
    partitioning_plan = partitioning_plans_[i];
  }
  else {
//    partitioning_plan = NULL ;  //Local Execution
    partitioning_plan = partitioning_plans_[0]; // Send data and wait, even if model not sent
  }

//  *(net_->offloaded_layers_.begin())=make_pair(offloaded_layers_.first, offloaded_layers_.second);

  /* Forward execution */
	gettimeofday(&start, NULL);


  net_->Forward(partitioning_plan);
  gettimeofday(&finish, NULL);
  timechk = (double)(finish.tv_sec) + (double)(finish.tv_usec) / 1000000.0 -
            (double)(start.tv_sec) - (double)(start.tv_usec) / 1000000.0;

  cout << "Forward time : " << timechk << " s" << endl;
  timechk2 = (double)(start.tv_sec) + (double)(start.tv_usec) / 1000000.0 -
            (double)(start_global.tv_sec) - (double)(start_global.tv_usec) / 1000000.0;
  forwardTimeFile << timechk2 <<", "<<  timechk<<", ";



  timeFile << "," << timechk;

//  list< pair<int, int> >::iterator i;
  //cout << "Partitioning points for time optimization (k : " << 1 - exp(-1 * num_offloading_ * K_) << ")" << endl;
//  cout << "Partitioning points (k : " << 1 - pow(K_, num_offloading_ / 5) << ")" << endl;
//  for (i = partitioning_plan.begin(); i != partitioning_plan.end(); ++i) {
//    cout << "(" << (*i).first << ", " << (*i).second << ") " << endl;
//    layerFile << ",\"(" << (*i).first << ", " << (*i).second << ")\"";
//  }


  if (partitioning_plan == NULL) {
    layerFile << ",\"(-1, -1)\"";
  }
  else {
    list< pair<int, int> >::iterator k;
    for (k = partitioning_plan->begin(); k != partitioning_plan->end(); ++k) {
      layerFile << ",\"(" << (*k).first << ", " << (*k).second << ")\"";
    }
  }
//  forwardTimeFile<<endl;

//  timeFile.close();
//  layerFile.close();
//  forwardTimeFile.close();

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

int main(int argc, char** argv) {
  if (argc != 14) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " mean.binaryproto labels.txt server_predict.txt client_predict.txt 80(Mbps) img.jpg 0.5(K_value) incremental time(opt_target) use_file IP_address" << std::endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  string model_file   = argv[1];
  string trained_file = argv[2];
  string mean_file    = argv[3];
  string label_file   = argv[4];
  string server_prediction_model_file = argv[5];
  string client_prediction_model_file = argv[6];
  string network_speed = argv[7];
  string k_value = argv[9];
  float k = strtof(k_value.c_str(), NULL);
  string incremental = argv[10];
  bool is_incremental = incremental == "incremental"; // <-> all_at_once
  string opt_target = argv[11];
  string use_file_s = argv[12];
  string ip_address = argv[13];
  bool use_file = use_file_s == "use_file";
  //string ip_address = argv[12];

  if (!is_incremental) {
    /* Upload all */
    network_speed = "999999";
  }

  /* Connect to the server */
  boost::asio::io_service io_service;
  tcp::socket s(io_service);
  tcp::resolver resolver(io_service);
  tcp::resolver::query query(tcp::v4(), ip_address, "7676");
  //tcp::resolver::query query(tcp::v4(), ip_address, "7675");
  tcp::resolver::iterator iter = resolver.resolve(query);
  boost::asio::connect(s, iter);

  Classifier classifier(model_file, trained_file, mean_file, label_file, server_prediction_model_file, client_prediction_model_file, strtof(network_speed.c_str(), NULL), &s, k, is_incremental, opt_target, use_file, ip_address);

  string file = argv[8];

  std::cout << "---------- Prediction for "
            << file << " ----------" << std::endl;

  cv::Mat img = cv::imread(file, -1);
  CHECK(!img.empty()) << "Unable to decode image " << file;

  std::ofstream timeFile("time_measurement.csv", std::ios::app);
  timeFile << "," << incremental;
  timeFile.close();
  std::ofstream layerFile("layer_measurement.csv", std::ios::app);
  layerFile << "," << incremental;
  layerFile.close();

  /* Start background uploading */

  classifier.StartClassify(img);
  classifier.StartBackgroundUpload();

  usleep(500000);
	gettimeofday(&start_global, NULL);
  cond.notify_all();

  // Join background uploading thread
  classifier.JoinBackgroundUpload();
  classifier.JoinStartClassify();

  timeFile.open("time_measurement.csv", std::ios::app);
  timeFile << std::endl;
  timeFile.close();
  layerFile.open("layer_measurement.csv", std::ios::app);
  layerFile << std::endl;
  layerFile.close();

  std::ofstream forwardTimeFile("forward_time.csv", std::ios::app);
  forwardTimeFile << endl;
  forwardTimeFile.close();

}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
