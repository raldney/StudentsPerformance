import java.io.File;
import java.util.Arrays;

import org.encog.ConsoleStatusReportable;
import org.encog.Encog;
import org.encog.ml.MLRegression;
import org.encog.ml.data.MLData;
import org.encog.ml.data.versatile.NormalizationHelper;
import org.encog.ml.data.versatile.VersatileMLDataSet;
import org.encog.ml.data.versatile.columns.ColumnDefinition;
import org.encog.ml.data.versatile.columns.ColumnType;
import org.encog.ml.data.versatile.missing.MeanMissingHandler;
import org.encog.ml.data.versatile.sources.CSVDataSource;
import org.encog.ml.data.versatile.sources.VersatileDataSource;
import org.encog.ml.factory.MLMethodFactory;
import org.encog.ml.model.EncogModel;
import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;

public class Main {

	public void run(String[] args) {
		try {
			File filename = new File("./res/data-set.csv");

			CSVFormat format = CSVFormat.DECIMAL_POINT;
			VersatileDataSource source = new CSVDataSource(filename, false, format);
			VersatileMLDataSet data = new VersatileMLDataSet(source);
			data.getNormHelper().setFormat(format);

			ColumnDefinition columnGender = data.defineSourceColumn("gender", 0, ColumnType.ordinal);
			columnGender.defineClass(new String[] {"M","F"});
			data.defineSourceColumn("NationalITy", 1, ColumnType.nominal);
			data.defineSourceColumn("PlaceofBirth", 2,ColumnType.nominal);
			ColumnDefinition columnStageId = data.defineSourceColumn("StageID", 3, ColumnType.ordinal);
			columnStageId.defineClass(new String[] {"lowerlevel","MiddleSchool","HighSchool"});
			ColumnDefinition columnGradeID = data.defineSourceColumn("GradeID", 4, ColumnType.ordinal);
			columnGradeID.defineClass(new String[] {"G-02","G-03","G-04","G-05","G-06","G-07","G-08","G-09","G-10","G-11","G-12"});
			ColumnDefinition columnSectionID = data.defineSourceColumn("SectionID", 5, ColumnType.nominal);
			columnSectionID.defineClass(new String[] {"A","B","C"});
			data.defineSourceColumn("Topic", 6, ColumnType.nominal);
			ColumnDefinition columnSemester = data.defineSourceColumn("Semester", 7, ColumnType.ordinal);
			columnSemester.defineClass(new String[] {"F","S"});
			data.defineSourceColumn("Relation", 8, ColumnType.nominal);
			data.defineSourceColumn("raisehands", 9, ColumnType.continuous);
			data.defineSourceColumn("VisITedResources", 10, ColumnType.continuous);
			data.defineSourceColumn("AnnoucementsView", 11, ColumnType.continuous);
			data.defineSourceColumn("Discussion", 12, ColumnType.continuous);
			ColumnDefinition columnParentAnsweringSurvey =  data.defineSourceColumn("ParentAnsweringSurvey", 13, ColumnType.ordinal);
			columnParentAnsweringSurvey.defineClass(new String[] {"Yes","No"});
			ColumnDefinition columnParentschoolSatisfaction = data.defineSourceColumn("ParentschoolSatisfaction", 14, ColumnType.ordinal);
			columnParentschoolSatisfaction.defineClass(new String[] {"Bad","Good"});
			ColumnDefinition columnStudentAbsenceDays = data.defineSourceColumn("StudentAbsenceDays", 15, ColumnType.ordinal);
			columnStudentAbsenceDays.defineClass(new String[] {"Under-7","Above-7"});
			ColumnDefinition columnClass = data.defineSourceColumn("Class", 16, ColumnType.ordinal);
			columnClass.defineClass(new String[] {"L","M","H"});

			data.getNormHelper().defineUnknownValue("?");
			data.getNormHelper().defineMissingHandler(columnClass, new MeanMissingHandler());

			data.analyze();

			data.defineSingleOutputOthersInput(columnClass);

			// Create feedforward neural network as the model type. MLMethodFactory.TYPE_FEEDFORWARD.
			// MLMethodFactory.SVM:  Support Vector Machine (SVM)
			// MLMethodFactory.TYPE_RBFNETWORK: RBF Neural Network
			// MLMethodFactor.TYPE_NEAT: NEAT Neural Network
			// MLMethodFactor.TYPE_PNN: Probabilistic Neural Network
			EncogModel model = new EncogModel(data);
			model.selectMethod(data, MLMethodFactory.TYPE_FEEDFORWARD);

			// Send any output to the console.
			model.setReport(new ConsoleStatusReportable());
			data.normalize();

			// Use a seed of 1001 so that we always use the same holdback and will get more consistent results.
			model.holdBackValidation(0.3, true, 1001);

			model.selectTrainingType(data);

			MLRegression bestMethod = (MLRegression)model.crossvalidate(10, true);

			System.out.println( "Training error: " + model.calculateError(bestMethod, model.getTrainingDataset()));
			System.out.println( "Validation error: " + model.calculateError(bestMethod, model.getValidationDataset()));

			NormalizationHelper helper = data.getNormHelper();
			System.out.println(helper.toString());

			System.out.println("Final model: " + bestMethod);

			
			ReadCSV csv = new ReadCSV(filename, false, format);
			String[] line = new String[17];
			MLData input = helper.allocateInputVector();

			while(csv.next()) {
				StringBuilder result = new StringBuilder();

				line[0] = csv.get(0);
				line[1] = csv.get(1);
				line[2] = csv.get(2);
				line[3] = csv.get(3);
				line[4] = csv.get(4);
				line[5] = csv.get(5);
				line[6] = csv.get(6);
				line[7] = csv.get(7);
				line[8] = csv.get(8);
				line[9] = csv.get(9);
				line[10] = csv.get(10);
				line[11] = csv.get(11);
				line[12] = csv.get(12);
				line[13] = csv.get(13);
				line[14] = csv.get(14);
				line[15] = csv.get(15);
				 line[16] = csv.get(16);

				String correct = csv.get(0);
				helper.normalizeInputVector(line,input.getData(),false);
				MLData output = bestMethod.compute(input);
				String predictedMPG = helper.denormalizeOutputVectorToString(output)[0];

				result.append(Arrays.toString(line));
				result.append(" -> predicted: ");
				result.append(predictedMPG);
				result.append("(correct: ");
				result.append(correct);
				result.append(")");

				System.out.println(result.toString());
			}

			Encog.getInstance().shutdown();

		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}

	public static void main(String[] args) {
		Main prg = new Main();
		prg.run(args);
	}
}
