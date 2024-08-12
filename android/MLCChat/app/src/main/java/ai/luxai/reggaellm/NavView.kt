package ai.luxai.reggaellm

import android.app.Application
import android.content.Context
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController

@ExperimentalMaterial3Api
@Composable
fun NavView() {
    val context = LocalContext.current
    val application = context.applicationContext as Application
    val appViewModel = AppViewModel(application = application, context = context)

    val navController = rememberNavController()
    NavHost(navController = navController, startDestination = "home") {
        composable("home") { StartView(navController, appViewModel) }
        composable("chat") { ChatView(navController, appViewModel.chatState) }
    }
}