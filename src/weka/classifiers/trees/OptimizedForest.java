/*  Implementation of OptimizedForest - "Optimizing the number of trees in a
    decision forest to discover a subforest with high ensemble accuracy using 
    a genetic algorithm" by Md Nasim Adnan and Md Zahidul Islam (2016). 
    Copyright (C) <2020>  <Charles Sturt University>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>. 
    
    Author contact details: 
    Name: Michael Furner
    Email: mfurner@csu.edu.au
    Location: 	School of Computing and Mathematics, Charles Sturt University,
    			Bathurst, NSW, Australia, 2795.
 */
package weka.classifiers.trees;

import java.io.Serializable;
import java.util.Collections;
import java.util.LinkedList;
import java.util.Random;
import java.util.Vector;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.SingleClassifierEnhancer;
import weka.classifiers.meta.Bagging;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.TechnicalInformation;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;

/**
 * <!-- globalinfo-start -->
 * Implementation of the Optimal Subforest algorithm "OptimizedForest", which 
 * was published in:<br>
 * <br>
 * Md Nasim Adnan and Md Zahidul Islam: Optimizing the number of trees in a 
 * decision forest to discover a subforest with high ensemble accuracy using 
 * a genetic algorithm In: Knowledge-Based Systems Vol 110, 2016<br>
 * <br>
 * This algorithm builds a decision forest and then works out an optimal 
 * subforest via Genetic Algorithm.
 * <!-- globalinfo-end -->
 * 
 * <!-- options-start --> Valid options are:
 * <p/>
 * 
 * <pre>
 * -S &lt;num&gt;
 *  Seed for random number generator.
 *  (default 1)
 * </pre>
 *  
 * <pre>
 * -I &lt;num&gt;
 *  Number of iterations for genetic algorithm.
 *  (default 20)
 * </pre>
 * 
 * <pre>
 * -P &lt;num&gt;
 *  Initial population size for genetic algorithm.
 *  (default 20)
 * </pre>
 * 
 * <pre>
 * -C &lt; RandomForest | Bagging &gt;
 *  Decision forest building method.
 *  (Default = RandomForest)
 * </pre>
 *
 * <!-- options-end -->
 *
 * @author Michael Furner
 * @version $Revision: 1.0.1$
 */
public class OptimizedForest extends AbstractClassifier {
    
    /**
     * For serialization.
     */
    private static final long serialVersionUID = 66432423423L;
    
    /**
     * Number of iterations for GA.
     */
    private int m_numberIterations = 20;
    
    /**
     * Initial population size for GA.
     */
    private int m_sizeOfPopulation = 20;
    
    /**
     * Seed for random number generator.
     */
    private int m_randomSeed = 1;
      
    
    /** Enum for holding different build statuses */
    enum BuildStatus {
        BS_BUILT, BS_UNBUILT, BS_NOTCOMPATIBLE, BS_ONEATTRIBUTE
    };
    
     /** Classification type: RandomForest */
    public static final int CT_RANDOMFOREST = 1;
    /** Classification type: Bagging */
    public static final int CT_BAGGING = 2;
    
    /** Tags for displaying classification types in the GUI. */
    public static final Tag[] TAGS_CT = {
        new Tag(CT_RANDOMFOREST, "RandomForest."),
        new Tag(CT_BAGGING, "Bagging."),
    };
    
    /** Type of decision forest to use to build the trees. */
    private int classificationType = CT_RANDOMFOREST;
    
    /** The build status of this OptimizedForest object */
    private BuildStatus buildStatus = BuildStatus.BS_UNBUILT;
    
    /** Population of chromosomes in GA. */
    private ChromosomeCollection population;
    
    /** Holds and allows access to the individual decision forest trees. */
    private Holder classifierHolder;
        
    /** Dataset (stored for testing fitness of chromosomes). */
    private Instances m_data;
    
    /** Random number generator. */
    private Random m_random;
    
    /** The final optimal subforest chromosome found by OptimizedForest. Only
     * set when the algorithm completes successfully.
     */
    private Chromosome finalBest = null;
    
    /**
     * Main method for testing this class.
     *
     * @param args should contain the following arguments: -t training file [-T
     * test file] [-c class index]
     * @throws java.lang.Exception
     */
    public static void main(String[] args) throws Exception {
        runClassifier(new OptimizedForest(), args);
    }

    /**
     * Default constructor
     */
    public OptimizedForest() {
        m_random = new Random(m_randomSeed);
    }
    
    /**
     * Builds decision forest and finds an optimal subforest through the genetic
     * algorithm.
     *
     * @param instances - data with which to build the classifier
     * @throws java.lang.Exception
     */
    @Override
    public void buildClassifier(Instances instances) throws Exception {
        
        getCapabilities().testWithFail(instances);
        
        
        if(m_random == null) {
            m_random = new Random(m_randomSeed);
        }
        
        if(instances.numAttributes() == 1) { //dataset with only one attribute
            buildStatus = BuildStatus.BS_ONEATTRIBUTE;
        }
        
        if(buildStatus != BuildStatus.BS_UNBUILT) {     
            J48 useless = new J48();
            useless.buildClassifier(instances);
            return;
        }
        
        /* Step 0: Setup and Build Decision Forest */
        if(classificationType == CT_RANDOMFOREST) {
            classifierHolder = new RandomForestHolder(instances);
        }
        else {
            classifierHolder = new BaggingHolder(instances);
        }
       
        m_data = instances;
        
        /* Step 1: Initial Population Selection */
        //we need to calculate the accuracy and the diversity of each tree
        //create the base chromosome (a chromosome that uses all trees)
        Chromosome baseChr = new Chromosome(classifierHolder);
        population = new ChromosomeCollection(baseChr);
        for(int i = 0; i < m_sizeOfPopulation; i++) {
            Chromosome thisChromosome = new Chromosome(baseChr, i);
            population.addChromosome(thisChromosome);
        }
        Chromosome chrSFBest = population.getBest(); //so far best chromosome
        
        /* Main Loop */
        for (int j = 0; j < m_numberIterations; j++ ) {
            Chromosome chrCurrBest = population.getBest();
            
            /* Step 2: Crossover and Elitist Operation */
            //copy current population
            ChromosomeCollection tempCurr = new ChromosomeCollection(population);
            ChromosomeCollection pMod = new ChromosomeCollection(baseChr);
            
            
            for(int k = 0; k < m_sizeOfPopulation/2; k++) {
                
                Chromosome chrB = tempCurr.getBest();
                tempCurr.remove(tempCurr.getBestIndex());
                
                Chromosome chrR = tempCurr.getRoulette();
                tempCurr.remove(tempCurr.getLastRouletteIndex());
                
                Chromosome[] offspring = chrB.crossover(chrR);
                pMod.addChromosome(offspring[0]);
                pMod.addChromosome(offspring[1]);
                                
                
            }
            
            //final part of step 2 is elitist operation
            chrSFBest = elitistOperation(chrSFBest, chrCurrBest, pMod);
            
            /* Step 3: Mutation and Elitist Operation */
            for(int k = 0; k < pMod.chromosomes.size(); k++) {
                int t = m_random.nextInt(classifierHolder.getClassifiers().length);
                pMod.bitFlipper(k, t);
            }
            
            chrSFBest = elitistOperation(chrSFBest, chrCurrBest, pMod);
            
            /* Step 4: Chromosome Selection for the Next Iteration */
            ChromosomeCollection pPool = population.union(pMod);
            population = pPool.getRoulettePop();
            
            
        }
                
        //there's no point doing this if its already performing as well as possible
        if(chrSFBest.accuracyOverDataset < 1.0) {
                
            /* Step 5: Rectification of So Far Best Chromosome */
            for(int i = 0; i < chrSFBest.chromosomeEncoding.length; i++) {
                if(chrSFBest.chromosomeEncoding[i] == 0) {
                    chrSFBest.bitFlipperWithChecker(i);
                }
            }

            for(int i = 0; i < chrSFBest.chromosomeEncoding.length; i++) {
                if(chrSFBest.chromosomeEncoding[i] == 1) {
                    chrSFBest.bitFlipperWithChecker(i);
                }
            }
            
        }
        
        finalBest = chrSFBest;
        
        buildStatus = BuildStatus.BS_BUILT;
        
    }
    
    /**
     * Passes the classification through to the optimal subforest
     *
     * @param instance - the instance to be classified
     * @return index for the predicted class value
     * @throws java.lang.Exception
     */
    @Override
    public double classifyInstance(Instance instance) throws Exception {

        if(buildStatus == BuildStatus.BS_BUILT)
            return finalBest.predictMajorityVoting(instance);
        throw new Exception("Not Built! Can't classify!");

    }
    
    /**
     * Passes the classification through to the optimal subforest
     *
     * @param instance - the instance to be classified
     * @return probability distribution for this instance's classification
     * @throws java.lang.Exception
     */
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {

        if(buildStatus == BuildStatus.BS_BUILT)
            return finalBest.distributionForInstance(instance);
        throw new Exception("Not Built! Can't classify!");

    }
    
    /**
     * Replace the worst chromosome in the pool with the current best chromosome
     * and work out if the best chromosome in the pool is better than the 
     * overall all-time best so far chromosome.
     * @param chrSFBest - Overall so far best chromosome
     * @param chrCurrBest - current best chromosome in pool 
     * @param pool - pool of chromosomes
     * @return new chrSFBest if it changes, chrSFBest argument if not
     * @throws Exception
     */
    public Chromosome elitistOperation(Chromosome chrSFBest, Chromosome chrCurrBest, ChromosomeCollection pool) throws Exception {
        
        Chromosome returnChromosome = chrSFBest;
        
        Chromosome chrModBest = pool.getBest();
        if(chrModBest.getAccuracyOverDataset() > chrSFBest.getAccuracyOverDataset()) {
            pool.replaceBest(chrModBest);
            returnChromosome = chrModBest;
        }
        
        //check if the current best is better than the worst
        Chromosome chrModWorst = pool.getWorst();
        if(chrCurrBest.getAccuracyOverDataset() > chrModWorst.getAccuracyOverDataset()) {
            pool.replaceWorst(chrCurrBest);
        }
        
        return returnChromosome;
        
    }
    
    /**
     * Return subforest string.
     * @return String version of trees selected in chromosome.
     */
    @Override
    public String toString() {
        String outString = "";
        if (buildStatus == BuildStatus.BS_BUILT) {
            outString += finalBest.toString();
        }
        else {
            if(buildStatus == BuildStatus.BS_NOTCOMPATIBLE) {
                outString = "OptimizedForest not built!\nWeka OptimizedForest can currently only parse RandomForest or Bagging.";
            }
            else if(buildStatus == BuildStatus.BS_ONEATTRIBUTE) {
                outString = "OptimizedForest not built!\nUse a dataset with more than one attribute.";
            }
        }
        return outString;
    }
    
    /**
     * Returns an instance of a TechnicalInformation object, containing detailed
     * information about the technical background of this class, e.g., paper
     * reference or book this class is based on.
     *
     * @return the technical information about this class
     */
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;

        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "Md Nasim Adnan & Md Zahidul Islam");
        result.setValue(TechnicalInformation.Field.YEAR, "2016");
        result.setValue(TechnicalInformation.Field.TITLE, "Optimizing the number of trees in a decision forest to discover a subforest with high ensemble accuracy using a genetic algorithm");
        result.setValue(TechnicalInformation.Field.JOURNAL, "Knowledge-Based Systems");
        result.setValue(TechnicalInformation.Field.PUBLISHER, "Elsevier");
        result.setValue(TechnicalInformation.Field.VOLUME, "110");
        result.setValue(TechnicalInformation.Field.PAGES, "86-97");
        result.setValue(TechnicalInformation.Field.URL, "https://doi.org/10.1016/j.knosys.2016.07.016");

        return result;

    }
    
    /**
     * Returns capabilities of algorithm
     *
     * @return Weka capabilities of SysFor
     */
    @Override
    public Capabilities getCapabilities() {

        Capabilities result = super.getCapabilities();   // returns the object from weka.classifiers.Classifier

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);
        result.disable(Capabilities.Capability.RELATIONAL_ATTRIBUTES);
        result.disable(Capabilities.Capability.STRING_ATTRIBUTES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.BINARY_CLASS);
        result.disable(Capabilities.Capability.NUMERIC_CLASS);
        result.disable(Capabilities.Capability.DATE_CLASS);
        result.disable(Capabilities.Capability.RELATIONAL_CLASS);
        result.disable(Capabilities.Capability.UNARY_CLASS);
        result.disable(Capabilities.Capability.NO_CLASS);
        result.disable(Capabilities.Capability.STRING_CLASS);
        return result;

    }
    
    /**
     * Return a description suitable for displaying in the
     * explorer/experimenter.
     *
     * @return a description suitable for displaying in the
     * explorer/experimenter
     */
    public String globalInfo() {
        return "Implementation of the OptimizedForest algorithm, using a Genetic Algorithm to determine an optimal subforest from RandomForest or Bagging: \n"
                + "Adnan, M. N., & Islam, M. Z. (2016). Optimizing the number of trees in a decision forest to discover a "
                + "subforest with high ensemble accuracy using a genetic algorithm.\n"
                + "For more information, see:\n\n" + getTechnicalInformation().toString();
    }
    
    private final class Chromosome implements Serializable, Cloneable {
        
        /**
        * For serialization.
        */
       private static final long serialVersionUID = 4432423423L;
        
        protected Holder decisionForest;
        protected TreeMetrics[] metricForEachTree;
        protected double averageAccuracy;
        protected double averageKappa;
        protected double stdAccuracy;
        protected double stdKappa;
        
        protected double accuracyOverDataset = Double.NaN;
        
        protected int[] chromosomeEncoding;
        
        //copy constructor
        public Chromosome(Chromosome other) {
            this.decisionForest = other.decisionForest;
            
            this.metricForEachTree = new TreeMetrics[other.metricForEachTree.length];
            System.arraycopy(other.metricForEachTree, 0, this.metricForEachTree, 0, other.metricForEachTree.length);
            
            this.chromosomeEncoding = new int[other.chromosomeEncoding.length];
            System.arraycopy(other.chromosomeEncoding, 0, this.chromosomeEncoding, 0, other.chromosomeEncoding.length);
            
            this.averageAccuracy = other.averageAccuracy;
            this.averageKappa = other.averageKappa;
            this.stdAccuracy = other.stdAccuracy;
            this.stdKappa = other.stdKappa;
            this.accuracyOverDataset = other.accuracyOverDataset;

        }
        
        //copy constructor that includes a new chromosome but the same forest
        public Chromosome(Chromosome other, int[] chromosomeEncoding) {
            this.decisionForest = other.decisionForest;
            
            this.metricForEachTree = new TreeMetrics[other.metricForEachTree.length];
            System.arraycopy(other.metricForEachTree, 0, this.metricForEachTree, 0, other.metricForEachTree.length);
            
            this.chromosomeEncoding = new int[chromosomeEncoding.length];
            System.arraycopy(chromosomeEncoding, 0, this.chromosomeEncoding, 0, chromosomeEncoding.length);
            
            this.averageAccuracy = other.averageAccuracy;
            this.averageKappa = other.averageKappa;
            this.stdAccuracy = other.stdAccuracy;
            this.stdKappa = other.stdKappa;

        }
        
        public Chromosome(Holder decisionForest) throws Exception {
            this.decisionForest = decisionForest;
            
            Classifier[] tempTrees = classifierHolder.getClassifiers();
            metricForEachTree = new TreeMetrics[tempTrees.length];
            for(int i = 0; i < tempTrees.length; i++) {
                metricForEachTree[i] = new TreeMetrics(tempTrees[i]);
            }
            
            getBaseChromosome();
            
        }
        
        public Chromosome(Chromosome base, int iteration) {
            this.decisionForest = base.decisionForest;
            
            this.metricForEachTree = new TreeMetrics[base.metricForEachTree.length];
            System.arraycopy(base.metricForEachTree, 0, this.metricForEachTree, 0, base.metricForEachTree.length);
            
            this.averageAccuracy = base.averageAccuracy;
            this.averageKappa = base.averageKappa;
            this.stdAccuracy = base.stdAccuracy;
            this.stdKappa = base.stdKappa;
            
            if(iteration % 2 == 0) {
                setChromosomeEncodingRandom(); 
            }
            else {
                setChromosomeEncodingStrata();
            }
            
            
        }
        
        //this base chromosome is where the kappa of each individual tree vs the whole forest is calculated
        public void getBaseChromosome() throws Exception {
            
            //set all trees to active for this base chromosome
            chromosomeEncoding = new int[decisionForest.getClassifiers().length];
            
            for(int i = 0; i < chromosomeEncoding.length; i++ ) {
                chromosomeEncoding[i] = 1;
            }
            
            //calculate the kappa of the trees
            for(int i = 0; i < chromosomeEncoding.length; i++) {
                
                //get this tree, we will be working out its kappa in relation to all the rest of the trees
                TreeMetrics thisTree = metricForEachTree[i];
                int[] thisTreePredictions = new int[m_data.numClasses()];
                
                //this creates another chromosome with all except this tree
                Chromosome allOther = allExcept(i);
                int[] allOtherPredictions = new int[m_data.numClasses()];
                
                int numberAgreements = 0;
                
                for(int n = 0; n < m_data.size(); n++) {
                    
                    int thisTreePred = (int)thisTree.classifyInstance(m_data.get(n));
                    int allOtherPred = (int)allOther.predictMajorityVoting(m_data.get(n));
                    
                    //build up the confusion matrix between these two classifiers
                    thisTreePredictions[thisTreePred]++;
                    
                    allOtherPredictions[allOtherPred]++;
                    
                    if(thisTreePred == allOtherPred) {
                        numberAgreements++;
                    }
                    
                }
                
                double kappaValue = kappa(thisTreePredictions, allOtherPredictions, numberAgreements);
                metricForEachTree[i].setKappa(kappaValue);
                //also set up the average kappa and accuracy
                averageKappa += kappaValue;
                averageAccuracy += metricForEachTree[i].getAccuracy();
                
            }
            
            averageKappa /= chromosomeEncoding.length;
            averageAccuracy /= chromosomeEncoding.length;
            
            //work out standard deviation
            for(int i=0; i < chromosomeEncoding.length; i++) {
               stdKappa += Math.pow( (metricForEachTree[i].getKappa() - averageKappa), 2);
               stdAccuracy += Math.pow( (metricForEachTree[i].getAccuracy()- averageAccuracy), 2);
            }
            stdKappa /= chromosomeEncoding.length;
            stdKappa = Math.sqrt(stdKappa);
            stdAccuracy /= chromosomeEncoding.length;
            stdAccuracy = Math.sqrt(stdAccuracy);
            
        }
        
        public double kappa(int[] classifierPreds, int[] groundTruthOrOtherClassifierPreds, int numAgreements) {
            
            int totalRecs = Utils.sum(classifierPreds);
            
            double observedAccuracy = (double)numAgreements / totalRecs;
            
            //calculate expected accuracy
            double expectedAccuracy = 0;
            for(int i = 0; i < classifierPreds.length; i++) {
                
                expectedAccuracy += (classifierPreds[i] * groundTruthOrOtherClassifierPreds[i]) / totalRecs;
                
            }
            expectedAccuracy /= totalRecs;
            
            double kappa = (observedAccuracy - expectedAccuracy) / (1.0 - expectedAccuracy);
            return kappa;
            
        }
        
        public void setChromosomeEncodingRandom() {
            
            int[] newChromosomeEncoding = new int[decisionForest.getClassifiers().length];
           
            int M = m_random.nextInt(newChromosomeEncoding.length) +1; //between 1 and |trees|
            
            for(int i = 0; i < M; i++) {
                
                int rand = m_random.nextInt(newChromosomeEncoding.length); //between 0 and |trees|-1
                
                if(newChromosomeEncoding[rand] == 1) {
                    i--; //decrement because we must do this again
                }
                else {
                    newChromosomeEncoding[rand] = 1;
                }
                
            }
            
            this.chromosomeEncoding = newChromosomeEncoding;
        }
        
        public void setChromosomeEncodingStrata() {
            
            int[] newChromosomeEncoding = new int[decisionForest.getClassifiers().length];
            
            //set up the three stratas: linked lists with the integer indexes of trees that fall in the specific strata
            LinkedList<Integer> strataOne = new LinkedList<>();
            LinkedList<Integer> strataTwo = new LinkedList<>();
            LinkedList<Integer> strataThree = new LinkedList<>();
            LinkedList<Integer> unStratad = new LinkedList<>();
            
            
            for(int i = 0; i < metricForEachTree.length; i++) {
                if(metricForEachTree[i].getAccuracy() > averageAccuracy 
                        && metricForEachTree[i].getKappa() < averageKappa) {
                    strataOne.add(i);
                }
                else if(metricForEachTree[i].getAccuracy() > (averageAccuracy - stdAccuracy)
                        && metricForEachTree[i].getKappa() < (averageKappa + stdKappa)) {
                    strataTwo.add(i);
                }
                else if(metricForEachTree[i].getAccuracy() > (averageAccuracy - (2*stdAccuracy))
                        && metricForEachTree[i].getKappa() < (averageKappa + (2*stdKappa))) {
                    strataThree.add(i);
                }
                else {
                    unStratad.add(i);
                }
            }
            
            int M = m_random.nextInt(metricForEachTree.length)+1; //between 1 and |trees|
            
            if(M < strataOne.size()) {
                
                for(int i = 0; i < M; i++) {
                    
                    int rand = m_random.nextInt(strataOne.size());
                    if(newChromosomeEncoding[strataOne.get(rand)] == 1) {
                        i--; //decrement as we must roll again
                    }
                    else {
                        newChromosomeEncoding[strataOne.get(rand)] = 1;
                    }
                    
                }
                
            }
            else if(M >= strataOne.size() && M < strataOne.size() + strataTwo.size()) {
                
                for(int i = 0; i < strataOne.size(); i++) {
                    newChromosomeEncoding[strataOne.get(i)] = 1; 
                }
                
                int mPrime = M - strataOne.size();
                
                for(int i = 0; i < mPrime; i++ ) {
                    int rand = m_random.nextInt(strataTwo.size());
                    if(newChromosomeEncoding[strataTwo.get(rand)] == 1) {
                        i--; //decrement as we must roll again
                    }
                    else {
                        newChromosomeEncoding[strataTwo.get(rand)] = 1;
                    }
                }
                
            }
            else if(M >= strataOne.size() + strataTwo.size()
                    && M < strataOne.size() + strataTwo.size() + strataThree.size()) {
                
                for(int i = 0; i < strataOne.size(); i++) {
                    newChromosomeEncoding[strataOne.get(i)] = 1; 
                }
                for(int i = 0; i < strataTwo.size(); i++) {
                    newChromosomeEncoding[strataTwo.get(i)] = 1; 
                }
                
                int mPrime = M - strataOne.size() - strataTwo.size();
                
                for(int i = 0; i < mPrime; i++ ) {
                    int rand = m_random.nextInt(strataThree.size());
                    if(newChromosomeEncoding[strataThree.get(rand)] == 1) {
                        i--; //decrement as we must roll again
                    }
                    else {
                        newChromosomeEncoding[strataThree.get(rand)] = 1;
                    }
                }
                
            }
            else {
                for(int i = 0; i < strataOne.size(); i++) {
                    newChromosomeEncoding[strataOne.get(i)] = 1; 
                }
                for(int i = 0; i < strataTwo.size(); i++) {
                    newChromosomeEncoding[strataTwo.get(i)] = 1; 
                }
                for(int i = 0; i < strataThree.size(); i++) {
                    newChromosomeEncoding[strataThree.get(i)] = 1; 
                }
                
                int mPrime = M - strataOne.size() - strataTwo.size() - strataThree.size();
                
                for(int i = 0; i < mPrime; i++ ) {
                    int rand = m_random.nextInt(unStratad.size());
                    if(newChromosomeEncoding[unStratad.get(rand)] == 1) {
                        i--; //decrement as we must roll again
                    }
                    else {
                        newChromosomeEncoding[unStratad.get(rand)] = 1;
                    }
                }
            }
            
            this.chromosomeEncoding = newChromosomeEncoding;
            
        }
        
        public String toString() {
            //TODO make it so that outputting the final subforest size is an option
            //TODO make RandomForest and Bagging options available to be modified
            String output = "";
            int sum = 0;
            for(int i = 0; i < chromosomeEncoding.length; i++) {
                if(chromosomeEncoding[i] == 1) { //this classifier is active
                    output += metricForEachTree[i].toString();
                    output += "\n";
                    sum++;
                }
            }
            output += "\nFinal subforest size: " + sum;
            
            return output;
        }
        
        public String toStringChromosomeEncoding() {
            String output = "";
            for(int i = 0; i < chromosomeEncoding.length; i++) {
                output += chromosomeEncoding[i];
                output += " ";
            }
            return output;
        }


        public Chromosome[] crossover(Chromosome other) {
            
             int[] otherCE = other.chromosomeEncoding;
             int[] thisCE = this.chromosomeEncoding;
             
             int[] leftCE = new int[this.chromosomeEncoding.length];
             int[] rightCE = new int[this.chromosomeEncoding.length];
             
             int splitPoint = m_random.nextInt(other.chromosomeEncoding.length);
             
             for(int i = 0; i < thisCE.length; i++) {
                 if(i <= splitPoint) {
                     leftCE[i] = thisCE[i];
                     rightCE[i] = otherCE[i];
                 }
                 else {
                     leftCE[i] = otherCE[i];
                     rightCE[i] = thisCE[i];
                 }
             }
             
             Chromosome[] ret = new Chromosome[2];
             ret[0] = new Chromosome(this, leftCE);
             ret[1] = new Chromosome(this, rightCE);
             
             return ret;
           
        }
        
        public void bitFlipper(int t) {
            if(chromosomeEncoding[t] == 0) {
                chromosomeEncoding[t] = 1;
            }
            else if (chromosomeEncoding[t] == 1) {
                chromosomeEncoding[t] = 0;
            }
            //set this oto NaN to recalculate accuracy next time it is checked
            accuracyOverDataset = Double.NaN;
            
        }
        
        public void bitFlipperWithChecker(int t) throws Exception {
            if(chromosomeEncoding[t] == 0) {
                chromosomeEncoding[t] = 1;
            }
            else if (chromosomeEncoding[t] == 1) {
                chromosomeEncoding[t] = 0;
            }
            
            double oldAccuracy = accuracyOverDataset;
            
            accuracyOverDataset = Double.NaN;
            double newAccuracy = getAccuracyOverDataset();
            
            if(oldAccuracy >= newAccuracy) {
                //convert the chromosome back to previous
                if(chromosomeEncoding[t] == 0) {
                    chromosomeEncoding[t] = 1;
                }
                else if (chromosomeEncoding[t] == 1) {
                    chromosomeEncoding[t] = 0;
                }
            
                accuracyOverDataset = oldAccuracy;
                
            }
            
        }
        
        public Chromosome allExcept(int i) {
            
            //copy over the array
            int[] newChromosomeEncoding = new int[chromosomeEncoding.length];
            System.arraycopy(chromosomeEncoding, 0, newChromosomeEncoding, 0, chromosomeEncoding.length);
            
            //set the given index to be unused
            newChromosomeEncoding[i] = 0;
            
            //use copy constructor to return new chromosome 
            Chromosome retChromosome = new Chromosome(this, newChromosomeEncoding);
            return retChromosome;
            
        }
        
        public double predictMajorityVoting(Instance instance) throws Exception {
            //iterate over the classifiers included in this chromosome and get their votes
            int[] combinedVote = new int[m_data.numClasses()];
                        
            for(int i = 0; i < chromosomeEncoding.length; i++) {
                if(chromosomeEncoding[i] == 1) { //this classifier is active
                    combinedVote[(int) metricForEachTree[i].classifyInstance(instance)]++;
                }
            }
            
            return (double) Utils.maxIndex(combinedVote);
            
        }
        
        public double[] distributionForInstance(Instance instance) throws Exception {
            //iterate over the classifiers included in this chromosome and get their votes
            double[] combinedVote = new double[m_data.numClasses()];
                        
            for(int i = 0; i < chromosomeEncoding.length; i++) {
                if(chromosomeEncoding[i] == 1) { //this classifier is active
                    combinedVote[(int) metricForEachTree[i].classifyInstance(instance)]++;
                }
            }
            
            Utils.normalize(combinedVote);
           
            return combinedVote;
            
        }
        
        public double getAccuracyOverDataset() throws Exception {
            
            //we only need to calculate this once, so if it is already calculated
            //we can just return it
            
            if(!Double.isNaN(accuracyOverDataset))
                return accuracyOverDataset;
            
            //calculate accuracy over dataset for this chromosome
            accuracyOverDataset = 0;
            
            for(int i = 0; i < m_data.numInstances(); i++) {
                double pred = this.predictMajorityVoting(m_data.get(i));
                double actual = m_data.get(i).classValue();
                
                if(pred == actual) {
                    accuracyOverDataset++;
                }
                
            }
            
            accuracyOverDataset /= m_data.size();
            
            return accuracyOverDataset;
            
            
        }
        
        
    }
    
    protected class RandomForestHolder extends RandomForest implements Serializable, Holder {
        
        /**
        * For serialization.
        */
       private static final long serialVersionUID = 2432423423L;
        
        RandomForestHolder(Instances instances) throws Exception {
            this.buildClassifier(instances);
        }

        @Override
        public Classifier[] getClassifiers() {
            return m_Classifiers;
        }
        
        
    }
    
    protected class BaggingHolder extends Bagging implements Serializable,Holder {
        
        /**
        * For serialization.
        */
       private static final long serialVersionUID = 5432423423L;
        
        BaggingHolder(Instances instances) throws Exception {
            this.buildClassifier(instances);
        }
        
        @Override
        public Classifier[] getClassifiers() {
            return m_Classifiers;
        }
        
    }
    
    protected class ChromosomeCollection implements Serializable {
        
        /**
        * For serialization.
        */
        private static final long serialVersionUID = 3432423423L;
        
        public Chromosome baseChromosome;
        
        protected LinkedList<Chromosome> chromosomes;
        
        protected LinkedList<Double> chromosomeAccuracies;
        protected double accuracySum = 0;
        protected double bestAccuracy = Double.NEGATIVE_INFINITY;
        protected int bestAccuracyIndex = -1;
        protected double worstAccuracy = Double.POSITIVE_INFINITY;
        protected int worstAccuracyIndex = -1;
        protected int lastRouletteIndex = -1;
              
        //copy constructor
        public ChromosomeCollection(ChromosomeCollection other) {
            this.baseChromosome = other.baseChromosome;
            this.bestAccuracy = other.bestAccuracy;
            this.bestAccuracyIndex = other.bestAccuracyIndex;
            this.worstAccuracy = other.worstAccuracy;
            this.worstAccuracyIndex = other.worstAccuracyIndex;
            this.accuracySum = other.accuracySum;
            
            this.chromosomes = new LinkedList<>();
            this.chromosomeAccuracies = new LinkedList<>();
            
            for(int i = 0; i < other.chromosomes.size(); i++) {
                this.chromosomes.add(other.chromosomes.get(i));
                this.chromosomeAccuracies.add(other.chromosomeAccuracies.get(i));
            }
                    
        }
        
        public ChromosomeCollection(Chromosome baseChromosome) {
            this.baseChromosome = baseChromosome;
            this.chromosomes = new LinkedList<>();
            this.chromosomeAccuracies = new LinkedList<>();
        }
        
        public void addChromosome(Chromosome c) throws Exception {
            
            if(Double.isNaN(c.accuracyOverDataset)) {
            
                double accuracy = 0;
                //work out the chromosome's accuracy
                for(int i = 0; i < m_data.size(); i++) {
                    double pred = c.predictMajorityVoting(m_data.get(i));
                    double actual = m_data.get(i).classValue();
                    if(pred == actual) {
                        accuracy++;
                    }
                }
                accuracy /= m_data.size();
                c.accuracyOverDataset = accuracy;
                
            }
            
            this.chromosomeAccuracies.add(c.accuracyOverDataset);
                
            this.chromosomes.add(c);
            accuracySum += c.accuracyOverDataset;
            
            if(c.accuracyOverDataset > bestAccuracy) {
                bestAccuracy = c.accuracyOverDataset;
                bestAccuracyIndex = this.chromosomeAccuracies.size()-1; //the most recent addition
            }
            if(c.accuracyOverDataset < worstAccuracy) {
                worstAccuracy = c.accuracyOverDataset;
                worstAccuracyIndex = this.chromosomeAccuracies.size()-1; //the most recent addition
            }
            
        }
        
        public ChromosomeCollection union(ChromosomeCollection other) throws Exception {
            
            ChromosomeCollection cloneOfThis = new ChromosomeCollection(this);
            
            for(int i = 0; i < other.chromosomes.size(); i++) {
                cloneOfThis.addChromosome(other.chromosomes.get(i));
            }
            
            return cloneOfThis;
            
        }
        
        public void remove(int remIdx) {
            
            chromosomes.remove(remIdx);
            accuracySum -= chromosomeAccuracies.remove(remIdx);
            
            if(!chromosomeAccuracies.isEmpty()) {
                if(bestAccuracyIndex == remIdx) {
                    //work out new best
                    bestAccuracy = Collections.max(chromosomeAccuracies);
                    bestAccuracyIndex = chromosomeAccuracies.indexOf(bestAccuracy);

                }
                else if(bestAccuracyIndex > remIdx) {
                    bestAccuracyIndex--;
                }

                if(worstAccuracyIndex == remIdx) {
                    //work out new best
                    worstAccuracy = Collections.min(chromosomeAccuracies);
                    worstAccuracyIndex = chromosomeAccuracies.indexOf(worstAccuracy);

                }
                else if(worstAccuracyIndex > remIdx) {
                    worstAccuracyIndex--;
                }

                if(lastRouletteIndex > remIdx) {
                    lastRouletteIndex--;
                }
            }
            else {
                bestAccuracyIndex = -1;
                worstAccuracyIndex = -1;
            }
            
            
        }
        
        public Chromosome getBest() {
            
            return this.chromosomes.get(bestAccuracyIndex);
            
        }
        
        public void replaceBest(Chromosome newChr) {
            
            this.chromosomes.set(bestAccuracyIndex, newChr);
            //we must also set the chromosome accuracy for this new chromosome
            this.chromosomeAccuracies.set(bestAccuracyIndex, newChr.accuracyOverDataset);
            
        }
        
        public Chromosome getWorst() {
            
            return this.chromosomes.get(worstAccuracyIndex);
            
        }
        
        public void replaceWorst(Chromosome newChr) {
            
            this.chromosomes.set(worstAccuracyIndex, newChr);
            //we must also set the chromosome accuracy for this new chromosome
            this.chromosomeAccuracies.set(worstAccuracyIndex, newChr.accuracyOverDataset);
            //and then we must work out the new worst chromosome
            
            worstAccuracy = Collections.min(this.chromosomeAccuracies);
            worstAccuracyIndex = this.chromosomeAccuracies.indexOf(worstAccuracy);
            
        }
        
        public int getBestIndex() {
            return bestAccuracyIndex;
        }
        public int getLastRouletteIndex() {
            return lastRouletteIndex;
        }
        
        public Chromosome getRoulette() {
            
            lastRouletteIndex = -1;
            
            if(chromosomeAccuracies.size() == 1) {
                lastRouletteIndex = 0;
            } 
            else {
                //normalise fitnesses
                double previousProbability = 0;
                double[] probabilities = new double[chromosomeAccuracies.size()];
                for(int i = 0; i < chromosomeAccuracies.size(); i++) {
                    probabilities[i] = previousProbability + (chromosomeAccuracies.get(i) / accuracySum);
                    previousProbability = probabilities[i];
                }

                while(lastRouletteIndex == -1) {
                    double roll = m_random.nextDouble();
                    for(int i = 0; i < probabilities.length; i++ ) {
                        double next = 1.0;
                        if(i != probabilities.length-1){
                            next = probabilities[i+1];
                        }

                        if(roll >= probabilities[i] && roll < next) {
                            lastRouletteIndex = i;
                        }
                    }
                }
            }
            
            return this.chromosomes.get(lastRouletteIndex);
            
        }
        
        public void bitFlipper(int chrIdx, int treeIdx) throws Exception {
            
            this.chromosomes.get(chrIdx).bitFlipper(treeIdx);
            
            double accuracy = 0;
            //work out the chromosome's accuracy
            for(int i = 0; i < m_data.size(); i++) {
                double pred = this.chromosomes.get(chrIdx).predictMajorityVoting(m_data.get(i));
                double actual = m_data.get(i).classValue();
                if(pred == actual) {
                    accuracy++;
                }
            }
            accuracy /= m_data.size();
            
            //replace the accuracy information for this newly altered chromosome
            double oldAccuracy = this.chromosomeAccuracies.get(chrIdx);
            this.chromosomeAccuracies.set(chrIdx, accuracy);
            accuracySum -= oldAccuracy;
            accuracySum += accuracy;

            this.chromosomes.get(chrIdx).accuracyOverDataset = accuracy;

            //update best and worst accuracy info
            if(accuracy > bestAccuracy) {
                bestAccuracy = accuracy;
                bestAccuracyIndex = this.chromosomeAccuracies.size()-1; //the most recent addition
            }
            if(accuracy < worstAccuracy) {
                worstAccuracy = accuracy;
                worstAccuracy = this.chromosomeAccuracies.size()-1; //the most recent addition
            }
            
            
        }
        
        public ChromosomeCollection getRoulettePop() throws Exception {
            
            ChromosomeCollection newPoppedPopn = new ChromosomeCollection(this.baseChromosome);
               
            //normalise fitnesses
            double previousProbability = 0;
            double[] probabilities = new double[chromosomeAccuracies.size()];
            for(int i = 0; i < chromosomeAccuracies.size(); i++) {
                probabilities[i] = previousProbability + (chromosomeAccuracies.get(i) / accuracySum);
                previousProbability = probabilities[i];
            }
            
            boolean[] selectedAlready = new boolean[chromosomeAccuracies.size()];
            
            while(newPoppedPopn.chromosomes.size() < m_sizeOfPopulation) {

                double roll = m_random.nextDouble();
                for(int i = 0; i < probabilities.length; i++ ) {
                    double next = 1.0;
                    if(i != probabilities.length-1){
                        next = probabilities[i+1];
                    }

                    if(roll >= probabilities[i] && roll < next) {
                        if(!selectedAlready[i]) {
                            newPoppedPopn.addChromosome(this.chromosomes.get(i));
                            selectedAlready[i] = true;
                        }
                    }
                }
                
            }
            
            return newPoppedPopn;
            
        }
        
    }
    
    private class TreeMetrics implements Serializable {
        
        /**
        * For serialization.
        */
       private static final long serialVersionUID = 1432423423L;
        
        protected Classifier classifier;
        
        protected double treeAccuracy;
        protected double treeKappa = Double.NaN;
        
        TreeMetrics(Classifier classifier) throws Exception {
            this.classifier = classifier;
            calculateAccuracy();
        }
        
        private void calculateAccuracy() throws Exception {
            //calculate accuracy by getting the vote for each instance from the
            //dataset
            int numberCorrect = 0;
            for(int i = 0; i < m_data.size(); i++) {
                
                double val = classifier.classifyInstance(m_data.get(i));
                if(val == m_data.get(i).classValue()) {
                    numberCorrect++;
                }
                
            }
            treeAccuracy = (double)numberCorrect / (double)m_data.size();
                        
        }
        
        public void setKappa(double k) {
            this.treeKappa = k;
        }
        
        public double getAccuracy() {
            return treeAccuracy;
        }
        
        public double getKappa() {
            return treeKappa;
        }
        
        public double classifyInstance(Instance instance) throws Exception {
            return classifier.classifyInstance(instance);
        }
        
        public String toString() {
            return classifier.toString();
        }
        
    }
    
    interface Holder {
        Classifier[] getClassifiers();
    }
    
    /**
     * Returns number of iterations for GA
     * @return number of iterations for GA
     */
    public int getNumberIterations() {
        return m_numberIterations;
    }

    /**
     * Sets number of iterations for GA
     * @param m_numberIterations - new number of iterations for GA
     */
    public void setNumberIterations(int m_numberIterations) {
        this.m_numberIterations = m_numberIterations;
    }
    
    /**
     * Return tip text for this option
     * @return tip text for this option
     */
    public String numberIterationsTipText() {
        return "Number of iterations for genetic algorithm.";
    }

    /**
     * Returns size of initial population (for GA)
     * @return size of initial population (for GA)
     */
    public int getSizeOfPopulation() {
        return m_sizeOfPopulation;
    }

    /**
     * Set the initial population size for GA.
     * @param m_sizeOfPopulation - new initial population size
     */
    public void setSizeOfPopulation(int m_sizeOfPopulation) {
        this.m_sizeOfPopulation = m_sizeOfPopulation;
    }
    
    /**
     * Return tip text for this option
     * @return tip text for this option
     */
    public String sizeOfPopulationTipText() {
        return "Size of initial population for genetic algorithm.";
    }
    
    /**
     * Set a new random number generator  seed value and create new Random object.
     * @param s - new seed value
     */
    public void setRandomSeed(int s) {
        m_randomSeed = s;
        m_random = new Random(s);
    }
    
    /**
     * Return random number generator seed.
     * @return current random number generator seed
     */
    public int getRandomSeed() {
        return m_randomSeed;
    }
    
    /**
     * Return tip text for this option
     * @return tip text for this option
     */
    public String randomSeedTipText() {
        return "Seed for random number generator.";
    }
    
    /**
     * Set sort method for displaying rules
     * @param newClassificationType
     */
    public void setClassificationType(SelectedTag newClassificationType) {
        if(newClassificationType.getTags() == TAGS_CT) {
            classificationType = newClassificationType.getSelectedTag().getID();
        }
    }
    
    /**
     * Return decision forest type
     * @return decision forest type
     */
    public SelectedTag getClassificationType() {
        return new SelectedTag(classificationType, TAGS_CT);
    }
    
    /**
     * Return tip text for this option
     * @return tip text for this option
     */
    public String classificationTypeTipText() {
        return "Type of decision forest to use.";
    }
    
    /**
     * Parse the options for OptimizedForest.
     * 
     * <!-- options-start --> Valid options are:
     * <p/>
     * 
     * <pre>
     * -S &lt;num&gt;
     *  Seed for random number generator.
     *  (default 1)
     * </pre>
     *  
     * <pre>
     * -I &lt;num&gt;
     *  Number of iterations for genetic algorithm.
     *  (default 20)
     * </pre>
     * 
     * <pre>
     * -P &lt;num&gt;
     *  Initial population size for genetic algorithm.
     *  (default 20)
     * </pre>
     * 
     * <pre>
     * -C &lt; RandomForest | Bagging &gt;
     *  Decision forest building method.
     *  (Default = RandomForest)
     * </pre>
     *
     * <!-- options-end -->
     * 
     * @param options
     * @throws Exception
     */
    @Override
    public void setOptions(String[] options) throws Exception {
        
        String sSeed = Utils.getOption('S', options);
        if (sSeed.length() != 0) {
            setRandomSeed(Integer.parseInt(sSeed));
        } else {
            setRandomSeed(1);
        }
        
        String sNumberIterations = Utils.getOption('I', options);
        if(sNumberIterations.length() != 0) {
            setNumberIterations(Integer.parseInt(sNumberIterations));
        }
        else {
            setNumberIterations(20);
        }
        
        String sPopulationSize = Utils.getOption('P', options);
        if(sPopulationSize.length() != 0) {
            setSizeOfPopulation(Integer.parseInt(sPopulationSize));
        }
        else {
            setSizeOfPopulation(20);
        }
        
        String sCT = Utils.getOption('C', options);
        if(sCT.length() != 0) {
            if(sCT.equals("RandomForest")) {
                setClassificationType(new SelectedTag(CT_RANDOMFOREST, TAGS_CT));
            }
            else if(sCT.equals("Bagging")) {
                setClassificationType(new SelectedTag(CT_BAGGING, TAGS_CT));
            }
            else {
                throw new IllegalArgumentException("Invalid sort method.");
            }
        }
       
        
        super.setOptions(options);
    }
    
    /**
     * Gets the current settings of the classifier.
     *
     * @return the current setting of the classifier
     */
    @Override
    public String[] getOptions() {
        Vector<String> result = new Vector<String>();
        
        result.add("-S");
        result.add("" + getRandomSeed());
        
        result.add("-I");
        result.add("" + getNumberIterations());
        
        result.add("-P");
        result.add("" + getSizeOfPopulation());
        
        result.add("-C");
        switch(classificationType) {
            case CT_RANDOMFOREST:
                result.add("RandomForest");
                break;
            case CT_BAGGING:
                result.add("Bagging");
                break;
        }
        

        Collections.addAll(result, super.getOptions());

        return result.toArray(new String[result.size()]);

    }
    
}
