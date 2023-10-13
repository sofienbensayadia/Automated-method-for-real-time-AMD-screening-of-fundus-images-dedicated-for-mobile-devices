package com.example.machineopencv;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.SVM;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.OptionalDouble;

import static java.lang.Math.abs;
import static org.opencv.core.CvType.CV_32FC1;
import static org.opencv.imgcodecs.Imgcodecs.CV_LOAD_IMAGE_COLOR;
import static org.opencv.imgproc.Imgproc.getRotationMatrix2D;
import static org.opencv.ml.Ml.ROW_SAMPLE;


public class MainActivity extends AppCompatActivity {
    TextView result;
    ImageView iv;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        result = (TextView) findViewById(R.id.tvResult);
        Button button = (Button) findViewById(R.id.button);
         iv = (ImageView) findViewById(R.id.iv);

        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                if (!OpenCVLoader.initDebug()) {
                    OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0, getApplicationContext(), baseLoaderCallback);
                } else {
                    baseLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);

                }

            }
        });
    }


    ////////////////////////////////////////////////////////////////////////////////
    BaseLoaderCallback baseLoaderCallback = new BaseLoaderCallback(this) {


        @Override
        public void onManagerConnected(int status) {
            super.onManagerConnected(status);
            if (status == LoaderCallbackInterface.SUCCESS) {
                try {

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                    float[] feature_im = new float[]{
                            1.1311836770003387f,0.5907635630016222f,0.06404030913358676f,
                            1.1374175813068828f,0.6500592079294767f,0.05893835066440951f,
                            1.1525110642396872f,0.5062716547519842f,0.060143416079110064f,
                            1.1360364049063316f,0.29928511907372585f,0.06310730509218951f,
                            1.1431111500409359f,0.714508135608087f,0.06661440775749422f,
                            1.1392641783027135f,0.9666023420025436f,0.057146652972016834f,
                            1.1694637343900005f,3.365334853734487f,0.1434512552708227f,
                            1.230082963771343f,1.2164598043945447f,0.12197761911807097f,
                            1.28318547810745f,1.9796500153502035f,0.13444015473642232f,
                            1.178279143745062f,0.6393798517608874f,0.10753783397793315f,
                            1.2394395148683688f,5.545677821148195f,0.10028128917613767f,
                            1.1984822901998218f,2.121858690408316f,0.17039621644388286f
                    };


//les labes de chaque image
                    int[] labels = new int[]{-1,-1,-1,-1,-1,-1, 1,1,1,1,1,1};

///  classificateur SVM

                    Mat trainingDataMat = new Mat(10, 3, CV_32FC1);
                    trainingDataMat.put(0, 0, feature_im);
                    Mat labelsMat = new Mat(10, 1, CvType.CV_32SC1);//
                    labelsMat.put(0, 0, labels);

                    SVM svm = SVM.create();
                   /* svm.trainAuto(trainingDataMat, ROW_SAMPLE, labelsMat);
                    double C = svm.getC();
                    double GAMMA = svm.getGamma();
                    int kernel = svm.getKernelType();

                    System.out.println ("kernetl Type, C and GAMMA are : " + kernel + ", " + C + ", " +GAMMA );*/

                    svm.setType(SVM.C_SVC);
                    svm.setKernel(SVM.RBF);
                    svm.setC(312.5);
                    svm.setGamma(0.5062500000000001);

                    svm.setTermCriteria(new TermCriteria(TermCriteria.MAX_ITER, 100, 1e-6));
                    svm.train(trainingDataMat, ROW_SAMPLE, labelsMat);
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                    //Instantiating the Imgcodecs class
                    Imgcodecs imageCodecs = new Imgcodecs();



                    Mat imga1;
                    imga1= Utils.loadResource(getApplicationContext(), R.drawable.sub03_test, CV_LOAD_IMAGE_COLOR);

                    Mat imga2;
                    imga2= Utils.loadResource(getApplicationContext(), R.drawable.sub08_test, CV_LOAD_IMAGE_COLOR);
                    Mat imga3;
                    imga3 = Utils.loadResource(getApplicationContext(), R.drawable.sub09_test, CV_LOAD_IMAGE_COLOR);
                    Mat imga4;
                    imga4 = Utils.loadResource(getApplicationContext(), R.drawable.sub16_test, CV_LOAD_IMAGE_COLOR);
                    Mat imga5;
                    imga5 = Utils.loadResource(getApplicationContext(), R.drawable.sub27_training, CV_LOAD_IMAGE_COLOR);
                    Mat imga6;
                    imga6 = Utils.loadResource(getApplicationContext(), R.drawable.sub28_training, CV_LOAD_IMAGE_COLOR);
                    Mat imga7;

                    imga7 = Utils.loadResource(getApplicationContext(), R.drawable.subimageexemple1, CV_LOAD_IMAGE_COLOR);
                    Mat imga8;
                    imga8 = Utils.loadResource(getApplicationContext(), R.drawable.subimageexemple2, CV_LOAD_IMAGE_COLOR);
                    Mat imga9;
                    imga9 = Utils.loadResource(getApplicationContext(), R.drawable.subimageexemple3, CV_LOAD_IMAGE_COLOR);
                    Mat imga10;
                    imga10 = Utils.loadResource(getApplicationContext(), R.drawable.subim0001, CV_LOAD_IMAGE_COLOR);
                    Mat imga11;
                    imga11 = Utils.loadResource(getApplicationContext(), R.drawable.subim0009, CV_LOAD_IMAGE_COLOR);
                    Mat imga12;
                    imga12 = Utils.loadResource(getApplicationContext(), R.drawable.subim0062, CV_LOAD_IMAGE_COLOR);
                    Mat imga13;
                    imga13 = Utils.loadResource(getApplicationContext(), R.drawable.subbehia, CV_LOAD_IMAGE_COLOR);
                    List< Mat> myList = new ArrayList< Mat >();
                    myList.add(imga1);
                    myList.add(imga2);
                    myList.add(imga3);
                    myList.add(imga4);
                    myList.add(imga5);
                    myList.add(imga6);
                    myList.add(imga7);
                    myList.add(imga8);
                    myList.add(imga9);
                    myList.add(imga10);
                    myList.add(imga11);
                    myList.add(imga12);
                    myList.add(imga13);

                    int[] data_labeltest = {-1,-1,-1,-1,-1,-1, 1,1,1,1,1,1,1};
                    Mat labelsMattest = new Mat(13, 1, CvType.CV_32SC1);//
                    labelsMattest.put(0, 0, data_labeltest);
                    ArrayList<Integer> res = new ArrayList<Integer>();
                    ArrayList<Integer> nodmla = new ArrayList<Integer>();
                    ArrayList<Integer> dmla = new ArrayList<Integer>();
                    for (int k = 0; k < 13; k++) {
                        Mat img;

                        img = myList.get(k);


                        Mat img2 = new Mat(img.height(), img.width(), img.type());
                        Imgproc.cvtColor(img, img2, Imgproc.COLOR_RGB2BGR);

                        if (img.dataAddr() == 0) {
                            System.out.println("Couldn't open file ");
                        } else {

                            System.out.println("Thresholding Example");

                        }
                        int hei = img.height();
                        int wid = img.width();
                        //      System.out.println("="+ hei +"\n");
                        List<Mat> bgr = new ArrayList<>();
                        Core.split(img2, bgr);

//////////////////////////////////////////////////////////////////////////////////////////

                        Mat img_green = bgr.get(1);

                  /*  Mat destination = new Mat(source.rows(), source.cols(), source.type());
                    Imgproc.equalizeHist(source, destination);
                    Imgcodecs.imwrite("enhancedParrot.jpg", destination);
                    int erosion_size = 5;
                    Mat element  = Imgproc.getStructuringElement(
                            Imgproc.MORPH_CROSS, new Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                            new Point(erosion_size, erosion_size)
                    );
                    Imgproc.erode(img, img, element);*/

/////////******image angle 0 et angle 90*//////////////

                        Mat im_rot_0 = imrotate(img_green, 0);


                        /////////******image angle 10 et angle 100*////////////

                        Mat im_rot_10 = imrotate(img_green, 10);


                        /////////******image angle 20 et angle 110*////////////

                        Mat im_rot_20 = imrotate(img_green, 20);


                        /////////******image angle 30 et angle 120*////////////

                        Mat im_rot_30 = imrotate(img_green, 30);


                        /////////******image angle 40 et angle 130*////////////

                        Mat im_rot_40 = imrotate(img_green, 40);


                        /////////******image angle 50 et angle 140*////////////

                        Mat im_rot_50 = imrotate(img_green, 50);


                        /////////******image angle 60 et angle 150*////////////

                        Mat im_rot_60 = imrotate(img_green, 60);


                        /////////******image angle 70 et angle 160*////////////

                        Mat im_rot_70 = imrotate(img_green, 70);
                        showImg(im_rot_70);

                        /////////******image angle 80 et angle 170*////////////

                        Mat im_rot_80 = imrotate(img_green, 80);

                        showImg(im_rot_80);


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        Mat PHoriz = new Mat(img.rows(), 9, CvType.CV_64F); // tableau de projection horizontale (0--85)


                        Mat Pvertic = new Mat(9, img.cols(), CvType.CV_64F); // tableau de projection horizontale (90--175)

/////////******image angle 0 et angle 90*//////////////
                        Core.reduce(im_rot_0, PHoriz.col(0), 1, Core.REDUCE_AVG, CvType.CV_64F);
                        Core.reduce(im_rot_0, Pvertic.row(0), 0, Core.REDUCE_AVG, CvType.CV_64F);
                        /////////******image angle 10 et angle 100*//////////////
                        Core.reduce(im_rot_10, PHoriz.col(1), 1, Core.REDUCE_AVG, CvType.CV_64F);
                        Core.reduce(im_rot_10, Pvertic.row(1), 0, Core.REDUCE_AVG, CvType.CV_64F);
                        /////////******image angle 20 et angle 110*//////////////
                        Core.reduce(im_rot_20, PHoriz.col(2), 1, Core.REDUCE_AVG, CvType.CV_64F);
                        Core.reduce(im_rot_20, Pvertic.row(2), 0, Core.REDUCE_AVG, CvType.CV_64F);
                        /////////******image angle 30 et angle 120*//////////////
                        Core.reduce(im_rot_30, PHoriz.col(3), 1, Core.REDUCE_AVG, CvType.CV_64F);
                        Core.reduce(im_rot_30, Pvertic.row(3), 0, Core.REDUCE_AVG, CvType.CV_64F);
                        /////////******image angle 40 et angle 130*//////////////
                        Core.reduce(im_rot_40, PHoriz.col(4), 1, Core.REDUCE_AVG, CvType.CV_64F);
                        Core.reduce(im_rot_40, Pvertic.row(4), 0, Core.REDUCE_AVG, CvType.CV_64F);
                        /////////******image angle 50 et angle 140*//////////////
                        Core.reduce(im_rot_50, PHoriz.col(5), 1, Core.REDUCE_AVG, CvType.CV_64F);
                        Core.reduce(im_rot_50, Pvertic.row(5), 0, Core.REDUCE_AVG, CvType.CV_64F);
                        /////////******image angle 60 et angle 150*//////////////
                        Core.reduce(im_rot_60, PHoriz.col(6), 1, Core.REDUCE_AVG, CvType.CV_64F);
                        Core.reduce(im_rot_60, Pvertic.row(6), 0, Core.REDUCE_AVG, CvType.CV_64F);
                        /////////******image angle 70 et angle 160*//////////////
                        Core.reduce(im_rot_70, PHoriz.col(7), 1, Core.REDUCE_AVG, CvType.CV_64F);
                        Core.reduce(im_rot_70, Pvertic.row(7), 0, Core.REDUCE_AVG, CvType.CV_64F);
                        /////////******image angle 80 et angle 170*//////////////
                        Core.reduce(im_rot_80, PHoriz.col(8), 1, Core.REDUCE_AVG, CvType.CV_64F);
                        Core.reduce(im_rot_80, Pvertic.row(8), 0, Core.REDUCE_AVG, CvType.CV_64F);
               /*  for (int X1 = 0; X1 < PHoriz.rows(); X1++) {
                    //  for (int Y1 = 0; Y1 < Pvertic.cols(); Y1++) {

                    System.out.println(PHoriz.get(X1,5)[0]);}//}
                    System.out.println("done");*/

                        // Mat MRADh;Mat MRADv;Mat MRAD ;
             /* if(PHoriz.rows()> Pvertic.cols() ){
                  MRADh = new  Mat(9,PHoriz.rows(), CvType.CV_64F );
                  MRADv = new  Mat(9,PHoriz.rows(), CvType.CV_64F );
                  MRAD =  new  Mat(18,PHoriz.rows(), CvType.CV_64F );
                  }else{
                  MRADv = new  Mat(9,Pvertic.cols(), CvType.CV_64F );
                  MRADh = new  Mat(9,Pvertic.rows(), CvType.CV_64F );
                  MRAD =  new  Mat(18,Pvertic.rows(), CvType.CV_64F );}*/

                        Core.transpose(PHoriz, PHoriz);

                        //  System.out.println(MRADh.rows());
                        //  System.out.println(MRADv.rows());
                        // Core.vconcat(MRADh,MRADv,MRAD);
                        // PHoriz.copyTo(MRADh);
                        //Pvertic.copyTo(MRADv);

                   /* for (int X1 = 0; X1 < MRADh.rows(); X1++) {
                       for (int Y1 = 0; Y1 < MRADh.cols(); Y1++) {
                           MRAD.put(0, 0, MRADh.get(X1, Y1));
                       }}
                    for (int X1 = 0; X1 < MRADv.rows(); X1++) {
                        for (int Y1 = 0; Y1 < MRADv.cols(); Y1++) {
                            MRAD.put(MRADh.rows(), 0, MRADv.get(X1, Y1));
                        }}

                   /* for (int X1 = 0; X1 < MRAD.rows(); X1++) {
                        for (int Y1 = 0; Y1 < MRAD.cols(); Y1++) {

                            System.out.println(""+X1+""+Y1+""+ MRAD.get(X1,Y1)[0]);
                        }}*/

                        double vmaxh = Core.minMaxLoc(PHoriz).maxVal;
                        double vminh = 100000;
                        double vmaxv = Core.minMaxLoc(Pvertic).maxVal;
                        double vminv = 100000;


                        for (int X1 = 0; X1 < PHoriz.rows(); X1++) {
                            for (int Y1 = 0; Y1 < PHoriz.cols(); Y1++) {
                                if ((vminh >= PHoriz.get(X1, Y1)[0]) && (PHoriz.get(X1, Y1)[0] != 0)) {
                                    vminh = PHoriz.get(X1, Y1)[0];
                                }

                            }
                        }
                        for (int X1 = 0; X1 < Pvertic.rows(); X1++) {
                            for (int Y1 = 0; Y1 < Pvertic.cols(); Y1++) {
                                if ((vminv >= Pvertic.get(X1, Y1)[0]) && (Pvertic.get(X1, Y1)[0] != 0)) {
                                    vminv = Pvertic.get(X1, Y1)[0];
                                }

                            }
                        }

                        double vmax = vmaxh > vmaxv ? vmaxh : vmaxv;
                        double vmin = vminh < vminv ? vminh : vminv;
                        double vsumtotal = 0;
                        double vsumintens = 0;
                        int nbint = 0;
                        int nbtt = 0;

                        for (int X1 = 0; X1 < Pvertic.rows(); X1++) {
                            for (int Y1 = 0; Y1 < Pvertic.cols(); Y1++) {
                                vsumtotal += Pvertic.get(X1, Y1)[0];
                                nbtt += 1;
                                if (Pvertic.get(X1, Y1)[0] >= ((vmax - vmin) * 0.9)) {
                                    vsumintens += Pvertic.get(X1, Y1)[0];
                                    nbint += 1;
                                }

                            }
                        }
                        //  System.out.println(nbint+"\t"+vsumintens);
                        // System.out.println(""+vsumtotal);
                        // System.out.println(""+ avgintens);
                        for (int X1 = 0; X1 < PHoriz.rows(); X1++) {
                            for (int Y1 = 0; Y1 < PHoriz.cols(); Y1++) {
                                vsumtotal += PHoriz.get(X1, Y1)[0];
                                nbtt += 1;
                                if (Pvertic.get(X1, Y1)[0] >= ((vmax - vmin) * 0.9)) {
                                    vsumintens += PHoriz.get(X1, Y1)[0];
                                    nbint += 1;
                                }
                            }
                        }
                        double avgintens = ((vsumintens / nbint) / (vsumtotal / nbtt));

                        //   System.out.println(nbint+"\t"+vsumintens);
                        //    System.out.println(""+vsumtotal);
                        System.out.println("" + avgintens);
                        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        Mat sum_Radhv;
                        Mat sum_Rad;
                        if (PHoriz.rows() > Pvertic.cols()) {

                            sum_Radhv = new Mat(2, PHoriz.cols(), CvType.CV_64F);
                            sum_Rad = new Mat(1, PHoriz.cols(), CvType.CV_64F);
                        } else {
                            sum_Radhv = new Mat(2, Pvertic.cols(), CvType.CV_64F);
                            sum_Rad = new Mat(1, Pvertic.cols(), CvType.CV_64F);
                        }
                        // Mat sum_Radhv =   new Mat(2,PHoriz.cols(), CvType.CV_64F);
                        //  Mat sum_Rad =   new Mat(1,PHoriz.cols(), CvType.CV_64F);
                        Core.reduce(PHoriz, sum_Radhv.row(0), 0, Core.REDUCE_SUM, CvType.CV_64F);
                        Core.reduce(Pvertic, sum_Radhv.row(1), 0, Core.REDUCE_SUM, CvType.CV_64F);
                        Core.reduce(sum_Radhv, sum_Rad.row(0), 0, Core.REDUCE_SUM, CvType.CV_64F);


                        for (int X1 = 0; X1 < sum_Rad.rows(); X1++) {
                            for (int Y1 = 0; Y1 < sum_Rad.cols(); Y1++) {
                                //  System.out.println("\t"+""+(sum_Rad.get(X1,Y1)[0]));
                                double A = (sum_Rad.get(X1, Y1)[0]);

                                sum_Rad.put(X1, Y1, (A / 18));
                                //  System.out.println("\t"+""+(sum_Rad.get(X1,Y1)[0]));
                            }
                        }

                        Mat di_mse;
                        if (PHoriz.rows() > Pvertic.cols()) {

                            di_mse = new Mat(1, PHoriz.cols(), CvType.CV_64F);

                        } else {
                            di_mse = new Mat(1, Pvertic.cols(), CvType.CV_64F);
                        }


                        for (int Y1 = 0; Y1 < di_mse.cols(); Y1++) {
                            di_mse.put(0, Y1, 0);
                        }


                        for (int X1 = 0; X1 < PHoriz.rows(); X1++) {
                            for (int Y1 = 0; Y1 < PHoriz.cols(); Y1++) {

                                //      System.out.println("\t"+""+(di_mse.get(X1,Y1)[0]));

                                double A1 = PHoriz.get(X1, Y1)[0];
                                double A = (sum_Rad.get(0, Y1)[0]);
                                double dff = abs(A1 - A);
                                double dff1 = dff + di_mse.get(0, Y1)[0];

                                di_mse.put(0, Y1, dff1);
                                //  System.out.println("\t"+""+( di_mse.get(0,Y1)[0]));
                            }
                        }
                        // System.out.println("\t"+""+sum_Rad.size()+""+PHoriz.cols()+""+Pvertic.cols());

                        for (int X1 = 0; X1 < Pvertic.rows(); X1++) {
                            for (int Y1 = 0; Y1 < Pvertic.cols(); Y1++) {

                                //      System.out.println("\t"+""+(di_mse.get(X1,Y1)[0]));
                                double A1 = Pvertic.get(X1, Y1)[0];
                                double A = (sum_Rad.get(0, Y1)[0]);
                                double dff = A1 - A;
                                double dff1 = dff + di_mse.get(0, Y1)[0];

                                di_mse.put(0, Y1, dff1);
                                //System.out.println("\t"+""+( di_mse.get(0,Y1)[0]));
                            }
                        }
                        double MS = 0;
                        for (int Y1 = 0; Y1 < di_mse.cols(); Y1++) {
                            double Sdf = di_mse.get(0, Y1)[0];
                            MS = MS + Sdf;
                        }

                        double MSE = MS / (18 * di_mse.cols());
                        System.out.println("\t" + "" + MSE);
                        /////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        double r = 0.2;

                        double value = 0;
                        int nbRT = 0;
                        double value1 = 0;
                        int nbRT1 = 0;
                        double value2 = 0;
                        int nbRT2 = 0;


                        for (int X1 = 0; X1 < PHoriz.rows(); X1++) {
                            double[] RT1 = new double[PHoriz.cols()];
                       /* for (int i = 0; i<RT1.length;++i){
                            RT1[i] = 0;
                        }*/
                            double[] entropy1 = new double[]{0};
                            for (int Y1 = 0; Y1 < PHoriz.cols(); Y1++) {
                                RT1[Y1] = PHoriz.get(X1, Y1)[0];
                            }

                            double sd = calculateSD(RT1);
                            entropy1 = sampleEntropy(RT1, r, sd, 1, 3);

                            OptionalDouble avgen = Arrays.stream(entropy1).average();
                            value1 = value1 + avgen.orElse(-1);
                            nbRT1 = nbRT1 + 1;
                            // System.out.println("sEv1"+ nbRT1+""+value1);

                   /* for (int Y1 = 0; Y1 <entropy1.length; Y1++) {
                        System.out.println("sEv1"+entropy1[Y1]);
                        }*/
                        }
/////////////////////////////////////////////////////////////////
                  /*  double sumval1=value;
                    double sumnbRT1=nbRT;
                    System.out.println("sEv1"+ sumval1);
                    System.out.println("sE1"+sumnbRT1);*/
                        /////////////////////////////////////////////
                        for (int X1 = 0; X1 < Pvertic.rows(); X1++) {
                            double[] RT2 = new double[Pvertic.cols()];
                            for (int i = 0; i < RT2.length; ++i) {
                                RT2[i] = 0;
                            }
                            double[] entropy2 = new double[]{0};
                            for (int Y1 = 0; Y1 < Pvertic.cols(); Y1++) {
                                RT2[Y1] = Pvertic.get(X1, Y1)[0];
                            }

                            double sd = calculateSD(RT2);
                            entropy2 = sampleEntropy(RT2, r, sd, 1, 3);
                            OptionalDouble max = Arrays.stream(entropy2).max();
                            //System.out.println("length"+ max);
                            value2 = value2 + max.orElse(-1);
                            //System.out.println("length"+ value);
                            nbRT2 = nbRT2 + 1;

                            //for (int Y1 = 0; Y1 <entropy.length; Y1++) {
                            //}
                        }
                  /*  double sumval2=value;
                    double sumnbRT2=nbRT;

                    System.out.println("sEv2"+ sumval2);
                    System.out.println("sE2x"+sumnbRT2);*/


                        double simpleentrop = (value1 + value2) / (nbRT1 + nbRT2);
                        System.out.println("sE" + simpleentrop);
///////////////////////////////////////////////////////////////////////////////////////////

                        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////// part test
                        float[] feature_im_test = {(float) avgintens, (float) MSE, (float) simpleentrop};

                        Mat sampleMat = new Mat(1, 3, CV_32FC1);
                        sampleMat.put(0, 0, feature_im_test);////////////

                        for (int X1 = 0; X1 < sampleMat.rows(); X1++) {

                            Mat testDataMat1 = new Mat(1, 3, CV_32FC1);
                            testDataMat1.put(0, 0, sampleMat.get(X1, 0));
                            testDataMat1.put(0, 1, sampleMat.get(X1, 1));
                            testDataMat1.put(0, 2, sampleMat.get(X1, 2));
                            int response = (int) svm.predict(testDataMat1);
                            res.add(response);
                            System.out.println(" :" + response);
                            //  if (response == -1)
                            //    nodmla.add(X1);//result.setText("DMLA NOT Detect");

                            //  else
                            //    result.setText("DMLA  Detected");
                        }
                    }
                   /* for(int i=0;i<res.size();i++){
                        System.out.println(" :"+res.get(i));
                    }*/

                    double tp = 0;
                    double fp = 0;
                    double tn = 0;
                    double fn = 0;
                    for (int i = 0; i < data_labeltest.length; i++) {
                        int p = res.get(i);
                        int  a = data_labeltest[i];
                        //	cout << predicted.at<int>(i, 0)<<"vrs"<< actual.at<int>(i, 0) <<"\t";
                        if (p > 0 && a > 0) {

                            ++tp;
                        }

                        else if (p < 0 && a < 0) {
                            ++tn;
                        }

                        else if (p < 0 && a > 0) {
                            ++fn;
                        }

                        else if (p > 0 && a < 0){
                            ++fp;
                        }
                    }

                    float Sensitivity = (float)(tp / (tp + fn));
                    float Specificity = (float)(tn/(tn + fp));
                    float Accuracy = (float)((tn + tp) / (tn+tp+fn+fp)) ;

                    System.out.println("Sensitivity="+Sensitivity+"\n");
                    System.out.println("Specificity="+Specificity+"\n");
                    System.out.println("Accuracy="+Accuracy+"\n");






























////////////////////////////////////////////////////////////////////////////////////://///////////////////////////////////////////////////////////////////
                } catch (Exception e) {
                    Log.e("error", "onManagerConnected: " + e.getMessage());
                    System.out.println("Unexcepted Exception");
                    e.printStackTrace();
                }
            }

        }

    };

    private void showImg(Mat img) {
        Bitmap bm = Bitmap.createBitmap(img.cols(), img.rows(),Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(img, bm);
        ImageView imageView = (ImageView) findViewById(R.id.iv);
        imageView.setImageBitmap(bm);
    }
    public Mat imrotate(Mat image, int angle) {
      //  Mat dst= new Mat();
      //  if (angle% 360.0 == 0.0)
      //  {
      ///      dst = image;
     //   }
      //  else{
        //Calculate size of new matrix
       // double radians = Math.toRadians(angle);
       // double sin = Math.abs(Math.sin(radians));
      //  double cos = Math.abs(Math.cos(radians));

      //  int newWidth = (int) (image.width() * cos + image.height() * sin);
      //  int newHeight = (int) (image.width() * sin + image.height() * cos);

        // rotating image
        //Point center = new Point(newWidth / 2, newHeight / 2);
        Point center = new Point(image.width() / 2, image.height() / 2);
        Mat rotMatrix = getRotationMatrix2D(center, angle, 1.0); //1.0 means 100 % scale

           /* Rect bbox =   new RotatedRect(center, image.size(), angle).boundingRect();

            // adjust transformation matrix
            rotMatrix.(0, 2) += bbox.width / 2.0 - center.x;
            rotMatrix.put(1, 2) += bbox.height / 2.0 - center.y;
*/

        Size size = new Size(image.width(), image.height());
        Imgproc.warpAffine(image, image, rotMatrix, image.size());//}

        return image;
        //return rotMatrix;
    }





       /* public static double getShannonEntropy(String s) {
            int n = 0;
            Map<Character, Integer> occ = new HashMap<>();

            for (int c_ = 0; c_ < s.length(); ++c_) {
                char cx = s.charAt(c_);
                if (occ.containsKey(cx)) {
                    occ.put(cx, occ.get(cx) + 1);
                } else {
                    occ.put(cx, 1);
                }
                ++n;
            }

            double e = 0.0;
            for (Map.Entry<Character, Integer> entry : occ.entrySet()) {
                char cx = entry.getKey();
                double p = (double) entry.getValue() / n;
                e += p * log2(p);
            }
            return -e;
        }

        private static double log2(double a) {
            return Math.log(a) / Math.log(2);
        }*/
    /**Calculate sample entropy
     @param	arr 	time series in (coarse-grained or otherwise)
     @param	r		tolerance (maximum distance)
     @param	sd		standard deviation of the original signal
     @param	tau		the length of the mean for the coarse-grained time series
     @param	mMax	the maximum m to calculate SE for
     @return se		sample entropies at m = 2 to mMax

     */
    private double[] sampleEntropy(double[] arr, double r, double sd, int tau, int mMax){
        double tolerance = r*sd;
        int[] cont = new int[mMax+2];
        for (int i = 0; i<cont.length;++i){
            cont[i] = 0;
        }
        for (int i = 0; i<arr.length-mMax;++i){
            for (int j = i+1; j<arr.length-mMax;++j){
                int k = 0;
                while (k < mMax && Math.abs(arr[i+k] - arr[j+k]) <= tolerance){
                    ++k;
                    cont[k]++;
                }
                if (k == mMax && Math.abs(arr[i+mMax] - arr[j+mMax]) <= tolerance)
                    cont[mMax+1]++;
            }
        }

        double[] se = new double[mMax];
        for (int i = 1; i <= mMax; ++i){
           if (cont[i+1] == 0 || cont[i] == 0){
                se[i-1] = -Math.log(1d/(((double) (arr.length-mMax))*(((double) (arr.length-mMax))-1d )));
            }else
            {
                se[i-1] = -Math.log(((double)cont[i+1])/((double)cont[i]));
            }
        }
        return se;
    }


    public static double calculateSD(double numArray[])
    {
        double sum = 0.0, standardDeviation = 0.0;
        int length = numArray.length;
        for(double num : numArray) {
            sum += num;
        }
        double mean = sum/length;
        for(double num: numArray) {
            standardDeviation += Math.pow(num - mean, 2);
        }
        return Math.sqrt(standardDeviation/length);
    }

}










